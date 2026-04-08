import { Plugin, PluginInput } from '@opencode-ai/plugin';
import { getConfigs, addConfig, removeConfig, LiteLLMConfig } from './config';
import { fetchModels, normalizeUrl, LiteLLMModel } from './client';
import { createConnectTool } from './tools';

function exposedID(m: LiteLLMModel): string {
  if (m.mode !== "responses") return m.id;
  if (!m.id.includes("/")) return m.id;
  if (!["chatgpt", "openai"].includes(m.litellmProvider || "")) return m.id;
  return m.id.split("/").slice(1).join("/");
}

function resolveModelID(m: LiteLLMModel, models: Record<string, any>): string {
  const id = exposedID(m);
  if (id === m.id) return id;
  if (models[id]) return m.id;
  return id;
}

function aliasFetch(aliases: Record<string, string>) {
  return async (input: RequestInfo | URL, init?: RequestInit) => {
    if (!init?.body || init.method !== "POST" || Object.keys(aliases).length === 0) {
      return fetch(input, init);
    }

    try {
      const body = JSON.parse(init.body as string);
      const alias = typeof body?.model === "string" ? aliases[body.model] : undefined;
      if (!alias) return fetch(input, init);

      return fetch(input, {
        ...init,
        body: JSON.stringify({
          ...body,
          model: alias,
        }),
      });
    } catch (_) {
      return fetch(input, init);
    }
  };
}

// Provider-specific reasoning effort levels
const REASONING_VARIANTS: Record<string, Record<string, any>> = {
  openai: {
    off:    { reasoningEffort: "off" },
    low:    { reasoningEffort: "low" },
    medium: { reasoningEffort: "medium" },
    high:   { reasoningEffort: "high" },
    xhigh:  { reasoningEffort: "xhigh" },
  },
  chatgpt: {
    off:    { reasoningEffort: "off" },
    low:    { reasoningEffort: "low" },
    medium: { reasoningEffort: "medium" },
    high:   { reasoningEffort: "high" },
    xhigh:  { reasoningEffort: "xhigh" },
  },
  gemini: {
    off:    { reasoningEffort: "off" },
    low:    { reasoningEffort: "low" },
    medium: { reasoningEffort: "medium" },
    high:   { reasoningEffort: "high" },
  },
  anthropic: {
    off:    { reasoningEffort: "off" },
    low:    { reasoningEffort: "low" },
    medium: { reasoningEffort: "medium" },
    high:   { reasoningEffort: "high" },
  },
  deepseek: {
    off:    { reasoningEffort: "off" },
    low:    { reasoningEffort: "low" },
    medium: { reasoningEffort: "medium" },
    high:   { reasoningEffort: "high" },
  },
};

const DEFAULT_REASONING_VARIANTS: Record<string, any> = {
  low:    { reasoningEffort: "low" },
  medium: { reasoningEffort: "medium" },
  high:   { reasoningEffort: "high" },
};

function getReasoningVariants(m: LiteLLMModel): Record<string, any> {
  if (m.litellmProvider && REASONING_VARIANTS[m.litellmProvider]) {
    return REASONING_VARIANTS[m.litellmProvider];
  }
  return DEFAULT_REASONING_VARIANTS;
}

/**
 * OpenCode LiteLLM provider plugin entry point.
 */
export const litellmPlugin: Plugin = async (_ctx: PluginInput) => {
  console.log('[litellm] Plugin initializing...');

  // Cache for fetched models to avoid redundant API calls
  const modelCache = new Map<string, LiteLLMModel[]>();

  return {
    /**
     * Config hook — dynamically registers providers and models from all
     * configured LiteLLM servers.
     */
    config: async (config: any) => {
      if (!config.provider) config.provider = {};

      const serverConfigs = getConfigs();

      for (const sc of serverConfigs) {
        const providerID = `litellm-${sc.alias}`;
        const baseUrl = normalizeUrl(sc.url);
        const aliases: Record<string, string> = {};

        // Fetch models if not cached
        let models: LiteLLMModel[] = modelCache.get(sc.alias) || [];
        if (models.length === 0) {
          try {
            console.log(`[litellm] Fetching models for ${sc.alias}...`);
            models = await fetchModels(sc.url, sc.key);
            modelCache.set(sc.alias, models);
          } catch (error) {
            console.error(`[litellm] Model fetch failed for ${sc.alias}:`, error);
          }
        }

        config.provider[providerID] = {
          id: providerID,
          name: `LiteLLM (${sc.alias})`,
          npm: "@ai-sdk/openai",
          api: "openai",
          options: {
            baseURL: `${baseUrl}/v1`,
            apiKey: sc.key,
            fetch: aliasFetch(aliases),
          },
          models: {},
        };

        if (models.length > 0) {
          for (const m of models) {
            const id = resolveModelID(m, config.provider[providerID].models);
            if (id !== m.id) aliases[id] = m.id;

            const modelConfig: any = {
              id,
              name: id === m.id ? (m.name || m.id) : m.id,
              limit: {
                context: m.contextWindow,
                output: m.maxOutputTokens,
              },
              capabilities: {
                temperature: true,
                reasoning: m.supportsReasoning,
                attachment: m.supportsVision,
                toolcall: m.supportsToolCalls,
                input: {
                  text: true,
                  image: m.supportsVision,
                  pdf: m.supportsVision,
                },
                output: { text: true },
              },
            };

            if (m.supportsReasoning) {
              modelConfig.variants = getReasoningVariants(m);
            }

            config.provider[providerID].models[id] = modelConfig;
          }
        } else {
          config.provider[providerID].models["placeholder"] = {
            id: "placeholder",
            name: "No models found (check connection)",
            limit: { context: 4096, output: 4096 },
          };
        }
      }

      console.log(`[litellm] Config hook completed. Registered ${serverConfigs.length} providers.`);
    },

    /**
     * Auth hook — handles initial provider setup and credential verification.
     * Management (list, remove) lives in the litellm:connect tool.
     */
    auth: {
      provider: "litellm",
      methods: [
        {
          type: "api" as const,
          label: "Connect to LiteLLM",
          prompts: [
            {
              type: "text" as const,
              key: "alias",
              message: "Alias for this server (e.g. 'work', 'staging')",
              placeholder: "my-server",
              validate(value: string) {
                if (!value || value.trim().length === 0) return "Alias is required";
                if (!/^[a-zA-Z0-9_-]+$/.test(value.trim()))
                  return "Alias must be alphanumeric (hyphens and underscores allowed)";
                return undefined;
              },
            },
            {
              type: "text" as const,
              key: "url",
              message: "LiteLLM base URL",
              placeholder: "https://litellm.example.com",
              validate(value: string) {
                if (!value || value.trim().length === 0) return "URL is required";
                try {
                  new URL(value.trim());
                } catch {
                  return "Must be a valid URL (e.g. https://litellm.example.com)";
                }
                return undefined;
              },
            },
            {
              type: "text" as const,
              key: "apiKey",
              message: "API key",
              placeholder: "sk-...",
              validate(value: string) {
                if (!value || value.trim().length === 0) return "API key is required";
                return undefined;
              },
            },
          ],
          async authorize(inputs?: Record<string, string>) {
            const alias = inputs?.alias?.trim();
            const url = inputs?.url?.trim();
            const apiKey = inputs?.apiKey?.trim();

            if (!alias || !url || !apiKey) {
              return { type: "failed" as const };
            }

            try {
              console.log(`[litellm] Verifying connection for ${alias} at ${url}...`);
              const models = await fetchModels(url, apiKey);
              console.log(`[litellm] Verified ${alias}: ${models.length} models available.`);

              // Persist to our config and cache
              modelCache.set(alias, models);
              addConfig({ alias, url, key: apiKey });

              return {
                type: "success" as const,
                key: apiKey,
                provider: `litellm-${alias}`,
              };
            } catch (error: any) {
              console.error(`[litellm] Verification failed for ${alias}:`, error.message);
              return { type: "failed" as const };
            }
          },
        },
      ],
    },

    /**
     * Register management tools.
     */
    tool: {
      'litellm_connect': createConnectTool(modelCache),
    },
  };
};
