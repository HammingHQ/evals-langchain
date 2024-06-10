import { BaseCallbackHandler } from "@langchain/core/callbacks/base";
import { Serialized } from "@langchain/core/load/serializable";
import { ChainValues } from "@langchain/core/utils/types";
import { LLMResult } from "@langchain/core/outputs";
import {
  BaseMessage,
  HumanMessage,
  ChatMessage,
  AIMessage,
  SystemMessage,
  FunctionMessage,
  ToolMessage,
} from "@langchain/core/messages";
import type { Document } from "@langchain/core/documents";
import {
  Hamming,
  MonitoringItem,
  GenerationParams,
  RetrievalParams,
} from "@hamming/hamming-sdk";

export class HammingCallbackHandler extends BaseCallbackHandler {
  name = "HammingCallbackHandler";
  private hamming: Hamming;
  private runItems: Record<string, MonitoringItem> = {};
  private runParent: Record<string, string> = {};
  private runLlmInput: Record<string, string> = {};
  private runLlmSerialized: Record<string, Serialized> = {};
  private runRetrieverQuery: Record<string, string> = {};
  private runRetrieverSerialized: Record<string, Serialized> = {};

  constructor(hamming: Hamming) {
    super();
    this.hamming = hamming;
    this.hamming.monitoring.start();
  }
  async handleChainStart(
    chain: Serialized,
    inputs: ChainValues,
    runId: string,
    parentRunId?: string | undefined,
    tags?: string[] | undefined,
    metadata?: Record<string, unknown> | undefined,
    runType?: string,
    name?: string,
  ): Promise<void> {
    try {
      if (parentRunId) {
        this.runParent[runId] = parentRunId;
        return;
      }
      const monitoringItem = await this.hamming.monitoring.startItem();
      monitoringItem.setInput(inputs);
      this.runItems[runId] = monitoringItem;
    } catch (e) {
      console.error("HammingCallbackHandler.handleChainStart", e);
    }
  }

  async handleChainEnd(
    outputs: ChainValues,
    runId: string,
    parentRunId?: string | undefined,
  ): Promise<void> {
    try {
      if (parentRunId) {
        delete this.runParent[runId];
        return;
      }
      const monitoringItem = this.runItems[runId];
      if (!monitoringItem) {
        console.warn("No monitoring item found for runId: ", runId);
        return;
      }
      delete this.runItems[runId];
      const obj = JSON.parse(JSON.stringify(outputs));
      monitoringItem.setOutput({
        response: obj.kwargs?.content,
      });
      monitoringItem.end();
    } catch (e) {
      console.error("HammingCallbackHandler.handleChainEnd", e);
    }
  }

  async handleChatModelStart(
    llm: Serialized,
    messages: BaseMessage[][],
    runId: string,
    parentRunId?: string,
    extraParams?: Record<string, unknown>,
    tags?: string[],
    metadata?: Record<string, unknown>,
    name?: string,
  ): Promise<void> {
    try {
      const flattenedMessages = messages.flat();
      const messageDicts = flattenedMessages.map((message) =>
        convertMessage(message),
      );
      this.runLlmInput[runId] = JSON.stringify(messageDicts);
      this.runLlmSerialized[runId] = llm;
    } catch (e) {
      console.error("HammingCallbackHandler.handleChatModelStart", e);
    }
  }

  async handleLLMStart(
    llm: Serialized,
    prompts: string[],
    runId: string,
    parentRunId?: string,
    extraParams?: Record<string, unknown>,
    tags?: string[],
    metadata?: Record<string, unknown>,
    name?: string,
  ): Promise<void> {
    try {
      this.runLlmInput[runId] = JSON.stringify(prompts);
      this.runLlmSerialized[runId] = llm;
    } catch (e) {
      console.error("HammingCallbackHandler.handleLLMStart", e);
    }
  }
  async handleLLMEnd(
    output: LLMResult,
    runId: string,
    parentRunId?: string,
    tags?: string[],
  ): Promise<void> {
    try {
      const llmInput = this.runLlmInput[runId];
      const llmSerialized = this.runLlmSerialized[runId];

      const lastGenArr = output.generations[output.generations.length - 1];
      const lastGen = lastGenArr[lastGenArr.length - 1];

      const params: GenerationParams = {
        input: llmInput,
        output: lastGen.text,
      };
      if (llmSerialized.id.includes("openai")) {
        params.metadata = {
          provider: "openai",
          model: llmSerialized["kwargs"]["model"],
          temperature: llmSerialized["kwargs"]["temperature"],
        };
      }

      if (parentRunId) {
        const topParentRunId = this._findTopParentRunId(parentRunId);
        const monitoringItem = this.runItems[topParentRunId];
        if (!monitoringItem) {
          console.warn("No monitoring item found for runId: ", topParentRunId);
          return;
        }
        monitoringItem.tracing.logGeneration(params);
      } else {
        this.hamming.tracing.logGeneration(params);
      }
    } catch (e) {
      console.error("HammingCallbackHandler.handleLLMEnd", e);
    }
  }

  async handleRetrieverStart(
    retriever: Serialized,
    query: string,
    runId: string,
    parentRunId?: string | undefined,
    tags?: string[] | undefined,
    metadata?: Record<string, unknown> | undefined,
    name?: string,
  ): Promise<void> {
    try {
      this.runRetrieverQuery[runId] = query;
      this.runRetrieverSerialized[runId] = retriever;
    } catch (e) {
      console.error("HammingCallbackHandler.handleRetrieverStart", e);
    }
  }

  async handleRetrieverEnd(
    documents: Document<Record<string, any>>[],
    runId: string,
    parentRunId?: string | undefined,
  ): Promise<void> {
    try {
      const query = this.runRetrieverQuery[runId];
      const serialized = this.runRetrieverSerialized[runId];
      const params: RetrievalParams = {
        query,
        results: documents.map((doc) => doc.pageContent),
        metadata: {
          engine: "langchain",
        },
      };
      if (parentRunId) {
        const topParentRunId = this._findTopParentRunId(parentRunId);
        const monitoringItem = this.runItems[topParentRunId];
        if (!monitoringItem) {
          console.warn("No monitoring item found for runId: ", topParentRunId);
          return;
        }
        monitoringItem.tracing.logRetrieval(params);
      } else {
        this.hamming.tracing.logRetrieval(params);
      }
    } catch (e) {
      console.error("HammingCallbackHandler.handleRetrieverEnd", e);
    }
  }

  _findTopParentRunId(runId: string): string {
    const parentRunId = this.runParent[runId];
    if (!parentRunId) {
      return runId;
    }
    return this._findTopParentRunId(parentRunId);
  }
}

function convertMessage(message: BaseMessage): Record<string, unknown> {
  let response: Record<string, unknown> | undefined;
  if (message instanceof HumanMessage) {
    response = { content: message.content, role: "user" };
  } else if (message instanceof ChatMessage) {
    response = { content: message.content, role: message.name };
  } else if (message instanceof AIMessage) {
    response = { content: message.content, role: "assistant" };
  } else if (message instanceof SystemMessage) {
    response = { content: message.content, role: "system" };
  } else if (message instanceof FunctionMessage) {
    response = {
      content: message.content,
      additional_kwargs: message.additional_kwargs,
      role: message.name,
    };
  } else if (message instanceof ToolMessage) {
    response = {
      content: message.content,
      additional_kwargs: message.additional_kwargs,
      role: message.name,
    };
  } else if (!message.name) {
    response = { content: message.content };
  } else {
    response = {
      role: message.name,
      content: message.content,
    };
  }
  return response;
}
