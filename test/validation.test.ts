import { expect, test, mock, describe, beforeEach } from "bun:test";
import { litellmPlugin } from "../src/index";

// Mocking the config module to return specified servers
mock.module("../src/config", () => {
  return {
    getConfigs: () => [
      { alias: "work", url: "http://server1", key: "key1" },
      { alias: "home", url: "http://server2", key: "key2" },
    ],
    addConfig: mock(() => {}),
    removeConfig: mock(() => {}),
  };
});

function mockFetch(handler: (url: string, init?: RequestInit) => Response) {
  globalThis.fetch = mock(async (input: any, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input.url;
    return handler(url, init);
  }) as any;
}

const modelsResponse = (models: { id: string }[]) =>
  new Response(JSON.stringify({ data: models }));

const modelInfoResponse = (entries: any[] = []) =>
  new Response(JSON.stringify({ data: entries }));

describe("LiteLLM Plugin Validation", () => {
  test("config hook registers providers and fetches models", async () => {
    mockFetch((url) => {
      if (url === "http://server1/v1/models")
        return modelsResponse([{ id: "model-a" }]);
      if (url === "http://server2/v1/models")
        return modelsResponse([{ id: "model-b" }]);
      if (url.endsWith("/model/info"))
        return modelInfoResponse();
      return modelsResponse([]);
    });

    const plugin = await litellmPlugin({} as any);
    const config: any = { provider: {} };
    await (plugin as any).config(config);

    expect(config.provider["litellm-work"]).toBeDefined();
    expect(config.provider["litellm-work"].models["model-a"]).toBeDefined();
    expect(config.provider["litellm-home"]).toBeDefined();
    expect(config.provider["litellm-home"].models["model-b"]).toBeDefined();
  });

  test("config hook sets conservative capabilities by default", async () => {
    mockFetch((url) => {
      if (url.endsWith("/v1/models"))
        return modelsResponse([{ id: "text-model" }]);
      if (url.endsWith("/model/info"))
        return modelInfoResponse();
      return modelsResponse([]);
    });

    const plugin = await litellmPlugin({} as any);
    const config: any = { provider: {} };
    await (plugin as any).config(config);

    const model = config.provider["litellm-work"].models["text-model"];
    expect(model.capabilities.reasoning).toBe(false);
    expect(model.capabilities.input.image).toBe(false);
    expect(model.capabilities.input.pdf).toBe(false);
    expect(model.capabilities.input.text).toBe(true);
    expect(model.capabilities.toolcall).toBe(true);
    expect(model.variants).toBeUndefined();
  });

  test("config hook enriches capabilities from /model/info", async () => {
    mockFetch((url) => {
      if (url.endsWith("/v1/models"))
        return modelsResponse([{ id: "gpt-4o" }]);
      if (url.endsWith("/model/info"))
        return modelInfoResponse([
          {
            model_name: "gpt-4o",
            model_info: {
              supports_vision: true,
              supports_function_calling: true,
              max_input_tokens: 128000,
              max_output_tokens: 16384,
            },
          },
        ]);
      return modelsResponse([]);
    });

    const plugin = await litellmPlugin({} as any);
    const config: any = { provider: {} };
    await (plugin as any).config(config);

    const model = config.provider["litellm-work"].models["gpt-4o"];
    expect(model.capabilities.input.image).toBe(true);
    expect(model.capabilities.toolcall).toBe(true);
    expect(model.capabilities.reasoning).toBe(false);
    expect(model.limit.context).toBe(128000);
    expect(model.limit.output).toBe(16384);
    expect(model.variants).toBeUndefined();
  });

  test("reasoning models get thinking variants", async () => {
    mockFetch((url) => {
      if (url.endsWith("/v1/models"))
        return modelsResponse([{ id: "o3" }, { id: "gpt-4o" }]);
      if (url.endsWith("/model/info"))
        return modelInfoResponse([
          {
            model_name: "o3",
            model_info: {
              supports_reasoning: null, // often null even for reasoning models
              supports_vision: true,
              supports_function_calling: true,
              max_input_tokens: 200000,
              max_output_tokens: 100000,
              litellm_provider: "openai",
              supported_openai_params: ["reasoning_effort", "temperature", "tools"],
            },
          },
          {
            model_name: "gpt-4o",
            model_info: {
              supports_reasoning: null,
              supports_vision: true,
              supports_function_calling: true,
              supported_openai_params: ["temperature", "tools"],
            },
          },
        ]);
      return modelsResponse([]);
    });

    const plugin = await litellmPlugin({} as any);
    const config: any = { provider: {} };
    await (plugin as any).config(config);

    const o3 = config.provider["litellm-work"].models["o3"];
    expect(o3.capabilities.reasoning).toBe(true);
    expect(o3.variants).toBeDefined();
    expect(o3.variants.off).toEqual({ reasoningEffort: "off" });
    expect(o3.variants.low).toEqual({ reasoningEffort: "low" });
    expect(o3.variants.medium).toEqual({ reasoningEffort: "medium" });
    expect(o3.variants.high).toEqual({ reasoningEffort: "high" });
    expect(o3.variants.xhigh).toEqual({ reasoningEffort: "xhigh" });

    const gpt4o = config.provider["litellm-work"].models["gpt-4o"];
    expect(gpt4o.capabilities.reasoning).toBe(false);
    expect(gpt4o.variants).toBeUndefined();
  });

  test("responses-mode models are exposed with OpenAI ids", async () => {
    mockFetch((url) => {
      if (url.endsWith("/v1/models"))
        return modelsResponse([{ id: "chatgpt/gpt-5.4" }, { id: "zai/glm-5" }]);
      if (url.endsWith("/model/info"))
        return modelInfoResponse([
          {
            model_name: "chatgpt/gpt-5.4",
            model_info: {
              mode: "responses",
              litellm_provider: "chatgpt",
              supports_function_calling: true,
              supported_openai_params: ["reasoning_effort", "tools"],
            },
          },
          {
            model_name: "zai/glm-5",
            model_info: {
              mode: "chat",
              supports_function_calling: true,
              supported_openai_params: ["tools"],
            },
          },
        ]);
      return modelsResponse([]);
    });

    const plugin = await litellmPlugin({} as any);
    const config: any = { provider: {} };
    await (plugin as any).config(config);

    expect(config.provider["litellm-work"].models["gpt-5.4"]).toBeDefined();
    expect(config.provider["litellm-work"].models["gpt-5.4"].name).toBe("chatgpt/gpt-5.4");
    expect(config.provider["litellm-work"].models["zai/glm-5"]).toBeDefined();
  });

  test("responses-mode aliases are rewritten back to LiteLLM ids", async () => {
    const calls: any[] = [];
    mockFetch((url, init) => {
      calls.push({ url, init });
      if (url.endsWith("/v1/models"))
        return modelsResponse([{ id: "chatgpt/gpt-5.4" }]);
      if (url.endsWith("/model/info"))
        return modelInfoResponse([
          {
            model_name: "chatgpt/gpt-5.4",
            model_info: {
              mode: "responses",
              litellm_provider: "chatgpt",
              supports_function_calling: true,
              supported_openai_params: ["reasoning_effort", "tools"],
            },
          },
        ]);
      return new Response(JSON.stringify({ ok: true }), { status: 200 });
    });

    const plugin = await litellmPlugin({} as any);
    const config: any = { provider: {} };
    await (plugin as any).config(config);

    await config.provider["litellm-work"].options.fetch("http://server1/v1/responses", {
      method: "POST",
      body: JSON.stringify({ model: "gpt-5.4", input: "hi" }),
      headers: { "content-type": "application/json" },
    });

    const body = JSON.parse(calls.at(-1).init.body);
    expect(body.model).toBe("chatgpt/gpt-5.4");
  });

  test("responses-mode aliases do not overwrite colliding models", async () => {
    mockFetch((url) => {
      if (url.endsWith("/v1/models"))
        return modelsResponse([{ id: "chatgpt/gpt-5.4" }, { id: "openai/gpt-5.4" }]);
      if (url.endsWith("/model/info"))
        return modelInfoResponse([
          {
            model_name: "chatgpt/gpt-5.4",
            model_info: {
              mode: "responses",
              litellm_provider: "chatgpt",
              supports_function_calling: true,
            },
          },
          {
            model_name: "openai/gpt-5.4",
            model_info: {
              mode: "responses",
              litellm_provider: "openai",
              supports_function_calling: true,
            },
          },
        ]);
      return modelsResponse([]);
    });

    const plugin = await litellmPlugin({} as any);
    const config: any = { provider: {} };
    await (plugin as any).config(config);

    expect(config.provider["litellm-work"].models["gpt-5.4"]).toBeDefined();
    expect(config.provider["litellm-work"].models["openai/gpt-5.4"]).toBeDefined();
  });

  test("auth hook returns SDK-compliant success with key", async () => {
    mockFetch((url) => {
      if (url.endsWith("/v1/models"))
        return modelsResponse([{ id: "test-model" }]);
      if (url.endsWith("/model/info"))
        return modelInfoResponse();
      return modelsResponse([]);
    });

    const plugin = await litellmPlugin({} as any);
    const authMethod = (plugin as any).auth.methods[0];

    expect(authMethod.label).toBe("Connect to LiteLLM");
    expect(authMethod.type).toBe("api");

    const result = await authMethod.authorize({
      alias: "test",
      url: "http://test-server",
      apiKey: "sk-test-key",
    });

    expect(result.type).toBe("success");
    expect(result.key).toBe("sk-test-key");
    expect(result.provider).toBe("litellm-test");
  });

  test("auth hook returns failed on bad connection", async () => {
    mockFetch(() => new Response("Unauthorized", { status: 401 }));

    const plugin = await litellmPlugin({} as any);
    const authMethod = (plugin as any).auth.methods[0];

    const result = await authMethod.authorize({
      alias: "bad",
      url: "http://bad-server",
      apiKey: "wrong-key",
    });

    expect(result.type).toBe("failed");
  });

  test("auth hook validates prompts", async () => {
    const plugin = await litellmPlugin({} as any);
    const prompts = (plugin as any).auth.methods[0].prompts;

    const aliasPrompt = prompts.find((p: any) => p.key === "alias");
    expect(aliasPrompt.validate("")).toBeDefined(); // required
    expect(aliasPrompt.validate("good-alias")).toBeUndefined(); // valid
    expect(aliasPrompt.validate("bad alias!")).toBeDefined(); // invalid chars

    const urlPrompt = prompts.find((p: any) => p.key === "url");
    expect(urlPrompt.validate("")).toBeDefined();
    expect(urlPrompt.validate("not-a-url")).toBeDefined();
    expect(urlPrompt.validate("https://litellm.example.com")).toBeUndefined();
  });

  test("connect tool add action works", async () => {
    mockFetch((url) => {
      if (url.endsWith("/v1/models"))
        return modelsResponse([{ id: "tool-model" }]);
      if (url.endsWith("/model/info"))
        return modelInfoResponse();
      return modelsResponse([]);
    });

    const plugin = await litellmPlugin({} as any);
    const tool = (plugin as any).tool["litellm_connect"];

    const result = await tool.execute({
      action: "add",
      alias: "tool-alias",
      url: "http://tool-url",
      key: "tool-key",
    });

    expect(result).toContain("Successfully connected");
  });

  test("connect tool list action works", async () => {
    // Need to initialize plugin (which calls no fetches for tool registration)
    mockFetch(() => modelsResponse([]));

    const plugin = await litellmPlugin({} as any);
    const tool = (plugin as any).tool["litellm_connect"];

    const result = await tool.execute({ action: "list" });
    expect(result).toContain("work");
    expect(result).toContain("home");
  });

  test("connect tool remove action works", async () => {
    mockFetch(() => modelsResponse([]));

    const plugin = await litellmPlugin({} as any);
    const tool = (plugin as any).tool["litellm_connect"];

    const result = await tool.execute({ action: "remove", alias: "work" });
    expect(result).toContain("Successfully removed");
  });
});
