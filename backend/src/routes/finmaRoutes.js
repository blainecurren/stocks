import express from "express";
import { FinMAMarketAnalyzer } from "../langchain/finmaAgent.js";
import dotenv from "dotenv";

dotenv.config();

const router = express.Router();
const analyzer = new FinMAMarketAnalyzer(
  process.env.DB_PATH || "./data/polygon_market_data.db",
  process.env.HUGGINGFACE_API_KEY
);

// Analyze individual stock
router.post("/api/finma/analyze-stock", async (req, res) => {
  try {
    const { ticker, date } = req.body;
    const analysis = await analyzer.analyzeStock(ticker, date);
    res.json({ success: true, analysis });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// Generate market report
router.get("/api/finma/market-report/:date", async (req, res) => {
  try {
    const { date } = req.params;
    const report = await analyzer.generateMarketReport(date);
    res.json({ success: true, report });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// Generate trading strategy
router.post("/api/finma/trading-strategy", async (req, res) => {
  try {
    const { tickers, date } = req.body;
    const strategy = await analyzer.generateTradingStrategy(tickers, date);
    res.json({ success: true, strategy });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// Batch analysis for multiple stocks
router.post("/api/finma/batch-analysis", async (req, res) => {
  try {
    const { tickers, date } = req.body;
    const analyses = [];

    for (const ticker of tickers) {
      const analysis = await analyzer.analyzeStock(ticker, date);
      analyses.push(analysis);
    }

    res.json({ success: true, analyses });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

export default router;
