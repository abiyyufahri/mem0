import { GoogleGenAI } from "@google/genai";
import { Embedder } from "./base";
import { EmbeddingConfig } from "../types";

export class GoogleEmbedder implements Embedder {
  private google: GoogleGenAI;
  private model: string;
  private outputDimensionality?: number;

  constructor(config: EmbeddingConfig) {
    this.google = new GoogleGenAI({ apiKey: config.apiKey });
    this.model = config.model || "text-embedding-004";
    // Configure vector size by passing `embeddingDims` directly in embedder config
    // or inside `modelProperties.embeddingDims`. If omitted, provider default is used.
    // Example: { embedder: { provider: 'google', config: { apiKey: 'KEY', embeddingDims: 512 }}}
    //          or { embedder: { provider: 'google', config: { modelProperties: { embeddingDims: 512 }}}}
    // NOTE: Google API will error if unsupported size is provided for the chosen model.
    // Prefer explicit embeddingDims, fall back to modelProperties if provided
    this.outputDimensionality = config.embeddingDims || config.modelProperties?.embeddingDims;
  }

  async embed(text: string): Promise<number[]> {
    const response = await this.google.models.embedContent({
      model: this.model,
      contents: text,
      // Only pass config if outputDimensionality is specified to avoid overriding provider defaults
      config: this.outputDimensionality
        ? { outputDimensionality: this.outputDimensionality }
        : undefined,
    });
    return response.embeddings![0].values!;
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    const response = await this.google.models.embedContent({
      model: this.model,
      contents: texts,
      config: this.outputDimensionality
        ? { outputDimensionality: this.outputDimensionality }
        : undefined,
    });
    return response.embeddings!.map((item: any) => item.values!);
  }
}
