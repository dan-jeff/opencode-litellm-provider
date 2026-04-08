export interface LiteLLMModel {
  id: string;
  name: string;
  mode: string | null;
  contextWindow: number;
  maxOutputTokens: number;
  supportsVision: boolean;
  supportsToolCalls: boolean;
  supportsReasoning: boolean;
  litellmProvider: string | null;
  supportedParams: string[];
}

export function normalizeUrl(url: string): string {
  let normalized = url.trim();
  if (normalized.endsWith('/')) {
    normalized = normalized.slice(0, -1);
  }
  if (normalized.endsWith('/v1')) {
    normalized = normalized.slice(0, -3);
  }
  return normalized;
}

/**
 * Fetches available models from the LiteLLM server via /v1/models,
 * then enriches with capability data from /model/info when available.
 */
export async function fetchModels(url: string, key: string): Promise<LiteLLMModel[]> {
  const baseUrl = normalizeUrl(url);
  const headers = {
    'Authorization': `Bearer ${key}`,
    'Content-Type': 'application/json',
  };

  // Fetch the model list
  const response = await fetch(`${baseUrl}/v1/models`, {
    method: 'GET',
    headers,
  });

  if (!response.ok) {
    let errorMessage = `Failed to fetch models: ${response.status} ${response.statusText}`;
    try {
      const errorData = await response.json();
      if (errorData?.error) {
        errorMessage += `. ${errorData.error.message || JSON.stringify(errorData.error)}`;
      }
    } catch (_) {}
    throw new Error(errorMessage);
  }

  const data = await response.json();
  if (!data || !Array.isArray(data.data)) {
    return [];
  }

  // Try to fetch detailed model info for capabilities
  let modelInfo: Record<string, any> = {};
  try {
    const infoResponse = await fetch(`${baseUrl}/model/info`, {
      method: 'GET',
      headers,
    });
    if (infoResponse.ok) {
      const infoData = await infoResponse.json();
      if (infoData?.data && Array.isArray(infoData.data)) {
        for (const entry of infoData.data) {
          const info = entry.model_info || {};
          const id = entry.model_name || info.id;
          if (id) modelInfo[id] = info;
        }
      }
    }
  } catch (_) {
    // /model/info is optional — fall back to conservative defaults
  }

  return data.data.map((model: any) => {
    const info = modelInfo[model.id] || {};
    const params: string[] = info.supported_openai_params || [];

    // supports_reasoning is often null even for reasoning models;
    // check if reasoning_effort is an accepted param as a stronger signal
    const reasoning = info.supports_reasoning === true
      || params.includes('reasoning_effort');

    return {
      id: model.id,
      name: model.name || model.id,
      mode: info.mode || null,
      contextWindow: info.max_input_tokens || info.max_tokens || model.context_window || 128000,
      maxOutputTokens: info.max_output_tokens || 4096,
      supportsVision: info.supports_vision ?? false,
      supportsToolCalls: info.supports_function_calling ?? true,
      supportsReasoning: reasoning,
      litellmProvider: info.litellm_provider || null,
      supportedParams: params,
    };
  });
}
