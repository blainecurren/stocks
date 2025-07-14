const axios = require("axios");

// Add axios to package.json dependencies first

// News analysis endpoint
app.post("/api/analyze-news", async (req, res) => {
  try {
    const { ticker, days_back, analysis_type } = req.body;

    // Call Python LLM service
    const response = await axios.post("http://localhost:8000/analyze_news", {
      ticker,
      days_back,
      analysis_type: analysis_type || "summary",
    });

    res.json(response.data);
  } catch (error) {
    console.error("Error calling LLM service:", error);
    res.status(500).json({
      error: "Failed to analyze news",
      details: error.message,
    });
  }
});

// Get analysis status
app.get("/api/llm-status", async (req, res) => {
  try {
    const response = await axios.get("http://localhost:8000/health");
    res.json(response.data);
  } catch (error) {
    res.status(503).json({
      status: "unavailable",
      error: error.message,
    });
  }
});
