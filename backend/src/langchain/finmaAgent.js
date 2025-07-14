import { HfInference } from "@huggingface/inference";
import { LLM } from "@langchain/core/language_models/llms";
import { PromptTemplate } from "@langchain/core/prompts";
import { LLMChain } from "langchain/chains";
import Database from "better-sqlite3";

export class FinMALLM extends LLM {
  constructor(fields) {
    super(fields);
    // FinMA-7B-full model
    this.model = fields.model || "ChanceFocus/finma-7b-full";
    this.hf = new HfInference(fields.apiKey);
    this.temperature = fields.temperature || 0.1;
    this.maxTokens = fields.maxTokens || 512;
  }

  _llmType() {
    return "finma";
  }

  async _call(prompt, options) {
    try {
      const response = await this.hf.textGeneration({
        model: this.model,
        inputs: prompt,
        parameters: {
          max_new_tokens: this.maxTokens,
          temperature: this.temperature,
          top_p: 0.95,
          do_sample: true,
          return_full_text: false,
        },
      });

      return response.generated_text;
    } catch (error) {
      console.error("FinMA API error:", error);
      throw error;
    }
  }
}

export class FinMAMarketAnalyzer {
  constructor(dbPath, hfApiKey) {
    this.dbPath = dbPath;

    // Initialize FinMA-7B
    this.llm = new FinMALLM({
      apiKey: hfApiKey,
      model: "ChanceFocus/finma-7b-full",
      temperature: 0.1,
      maxTokens: 1024,
    });

    this.db = new Database(dbPath, { readonly: true });
  }

  async analyzeStock(ticker, date) {
    // Get comprehensive stock data
    const stockData = this.db
      .prepare(
        `
      SELECT s.*, t.name, t.type 
      FROM snapshots_${date} s
      JOIN tickers t ON s.ticker = t.ticker
      WHERE s.ticker = ?
    `
      )
      .get(ticker);

    if (!stockData) {
      return { error: `No data found for ${ticker} on ${date}` };
    }

    // Get recent news
    const news = this.db
      .prepare(
        `
      SELECT title, description 
      FROM news_${date} 
      WHERE tickers LIKE ?
      ORDER BY published_utc DESC
      LIMIT 5
    `
      )
      .all(`%"${ticker}"%`);

    // FinMA prompt - structured for financial analysis
    const prompt = PromptTemplate.fromTemplate(`
      [INST] You are FinMA, a financial market analysis expert. Analyze the following stock data and provide professional investment insights.

      Stock: {ticker} - {name}
      Date: {date}
      
      Price Data:
      - Current: ${stockData.close}
      - Open: ${stockData.open}
      - High: ${stockData.high}
      - Low: ${stockData.low}
      - Previous Close: ${stockData.prev_close}
      - Change: {changePercent}%
      
      Volume:
      - Current: {volume}
      - Previous: {prevVolume}
      - Volume Change: {volumeChange}%
      
      Recent News:
      {newsText}
      
      Provide a comprehensive analysis including:
      1. Technical Analysis (support/resistance levels, trend direction)
      2. Volume Analysis (unusual activity, accumulation/distribution)
      3. Risk Assessment (volatility, market conditions)
      4. Trading Recommendation (buy/hold/sell with confidence level)
      5. Price Targets (near-term and medium-term)
      6. Key Risks to Monitor
      
      Format your response as a structured JSON object. [/INST]
    `);

    const newsText =
      news.length > 0
        ? news.map((n) => `- ${n.title}`).join("\n")
        : "No recent news available";

    const volumeChange =
      stockData.prev_volume > 0
        ? (
            ((stockData.volume - stockData.prev_volume) /
              stockData.prev_volume) *
            100
          ).toFixed(2)
        : 0;

    const chain = new LLMChain({ llm: this.llm, prompt });

    const response = await chain.call({
      ticker: stockData.ticker,
      name: stockData.name,
      date,
      changePercent: stockData.change_percent?.toFixed(2) || "0",
      volume: stockData.volume?.toLocaleString() || "0",
      prevVolume: stockData.prev_volume?.toLocaleString() || "0",
      volumeChange,
      newsText,
    });

    try {
      return JSON.parse(response.text);
    } catch (e) {
      // If JSON parsing fails, return structured response
      return {
        ticker,
        analysis: response.text,
        timestamp: new Date().toISOString(),
      };
    }
  }

  async generateMarketReport(date) {
    // Get market overview data
    const marketStats = this.db
      .prepare(
        `
      SELECT 
        COUNT(*) as total_stocks,
        SUM(CASE WHEN change_percent > 0 THEN 1 ELSE 0 END) as gainers,
        SUM(CASE WHEN change_percent < 0 THEN 1 ELSE 0 END) as losers,
        AVG(change_percent) as avg_change,
        SUM(volume) as total_volume
      FROM snapshots_${date}
    `
      )
      .get();

    // Get top movers
    const topGainers = this.db
      .prepare(
        `
      SELECT s.ticker, t.name, s.change_percent, s.volume
      FROM snapshots_${date} s
      JOIN tickers t ON s.ticker = t.ticker
      WHERE s.change_percent > 0
      ORDER BY s.change_percent DESC
      LIMIT 5
    `
      )
      .all();

    const topLosers = this.db
      .prepare(
        `
      SELECT s.ticker, t.name, s.change_percent, s.volume
      FROM snapshots_${date} s
      JOIN tickers t ON s.ticker = t.ticker
      WHERE s.change_percent < 0
      ORDER BY s.change_percent ASC
      LIMIT 5
    `
      )
      .all();

    const prompt = PromptTemplate.fromTemplate(`
      [INST] Generate a professional market analysis report using the following data:

      Market Date: {date}
      
      Market Overview:
      - Total Stocks Tracked: {totalStocks}
      - Gainers: {gainers} ({gainerPercent}%)
      - Losers: {losers} ({loserPercent}%)
      - Average Change: {avgChange}%
      - Total Volume: {totalVolume}
      
      Top Gainers:
      {topGainers}
      
      Top Losers:
      {topLosers}
      
      Create a comprehensive market report including:
      1. Executive Summary
      2. Market Sentiment Analysis
      3. Sector Performance Insights
      4. Volume Analysis
      5. Risk Indicators
      6. Trading Opportunities
      7. Market Outlook
      
      Use professional financial language and provide actionable insights. [/INST]
    `);

    const chain = new LLMChain({ llm: this.llm, prompt });

    const response = await chain.call({
      date,
      totalStocks: marketStats.total_stocks,
      gainers: marketStats.gainers,
      losers: marketStats.losers,
      gainerPercent: (
        (marketStats.gainers / marketStats.total_stocks) *
        100
      ).toFixed(1),
      loserPercent: (
        (marketStats.losers / marketStats.total_stocks) *
        100
      ).toFixed(1),
      avgChange: marketStats.avg_change?.toFixed(2) || "0",
      totalVolume: marketStats.total_volume?.toLocaleString() || "0",
      topGainers: topGainers
        .map((s) => `${s.ticker} (${s.name}): +${s.change_percent.toFixed(2)}%`)
        .join("\n"),
      topLosers: topLosers
        .map((s) => `${s.ticker} (${s.name}): ${s.change_percent.toFixed(2)}%`)
        .join("\n"),
    });

    return response.text;
  }

  async generateTradingStrategy(tickers, date) {
    // Get data for multiple tickers
    const tickerList = Array.isArray(tickers) ? tickers : [tickers];
    const stocksData = [];

    for (const ticker of tickerList) {
      const data = this.db
        .prepare(
          `
        SELECT s.*, t.name 
        FROM snapshots_${date} s
        JOIN tickers t ON s.ticker = t.ticker
        WHERE s.ticker = ?
      `
        )
        .get(ticker);

      if (data) stocksData.push(data);
    }

    const prompt = PromptTemplate.fromTemplate(`
      [INST] As FinMA, create a portfolio trading strategy for the following stocks:

      Portfolio Stocks:
      {stocksDetails}
      
      Market Date: {date}
      
      Develop a comprehensive trading strategy including:
      1. Portfolio Allocation Recommendations (% for each stock)
      2. Entry Points and Timing
      3. Stop Loss Levels
      4. Take Profit Targets
      5. Risk Management Rules
      6. Hedging Suggestions
      7. Time Horizon (day trade, swing, position)
      8. Overall Portfolio Risk Score
      
      Consider correlation, volatility, and market conditions. 
      Provide specific, actionable recommendations. [/INST]
    `);

    const stocksDetails = stocksData
      .map(
        (s) =>
          `${s.ticker} (${s.name}): Price $${
            s.close
          }, Change ${s.change_percent?.toFixed(
            2
          )}%, Volume ${s.volume?.toLocaleString()}`
      )
      .join("\n");

    const chain = new LLMChain({ llm: this.llm, prompt });
    const response = await chain.call({ stocksDetails, date });

    return response.text;
  }

  close() {
    this.db.close();
  }
}
