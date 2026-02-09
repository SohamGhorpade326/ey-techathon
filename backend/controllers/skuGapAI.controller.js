import SkuGap from "../models/SkuGap.js";
import axios from "axios";

/**
 * Helper function to fetch summary data
 */
async function fetchSummary() {
  const summary = await SkuGap.aggregate([
    {
      $facet: {
        totalItems: [{ $count: "count" }],
        avgMatch: [
          {
            $group: {
              _id: null,
              avg: { $avg: "$spec_match_percent" },
            },
          },
        ],
        fullCount: [
          { $match: { spec_match_percent: { $gte: 80 } } },
          { $count: "count" },
        ],
        partialCount: [
          {
            $match: {
              spec_match_percent: { $gte: 60, $lt: 80 },
            },
          },
          { $count: "count" },
        ],
        incompleteCount: [
          { $match: { spec_match_percent: { $lt: 60 } } },
          { $count: "count" },
        ],
        totalMissingSpecs: [
          {
            $group: {
              _id: null,
              total: { $sum: { $size: { $ifNull: ["$missing_specs", []] } } },
            },
          },
        ],
      },
    },
  ]);

  const result = summary[0];
  return {
    total_items: result.totalItems[0]?.count || 0,
    avg_match_percent: Math.round((result.avgMatch[0]?.avg || 0) * 100) / 100,
    full_count: result.fullCount[0]?.count || 0,
    partial_count: result.partialCount[0]?.count || 0,
    incomplete_count: result.incompleteCount[0]?.count || 0,
    total_missing_specs: result.totalMissingSpecs[0]?.total || 0,
  };
}

/**
 * Helper function to fetch domain analysis
 */
async function fetchDomainAnalysis() {
  const analysis = await SkuGap.aggregate([
    {
      $group: {
        _id: "$domain",
        total_gaps: { $sum: { $size: { $ifNull: ["$missing_specs", []] } } },
        avg_match: { $avg: "$spec_match_percent" },
      },
    },
    {
      $project: {
        _id: 0,
        domain: "$_id",
        total_gaps: 1,
        avg_match: { $round: ["$avg_match", 2] },
      },
    },
    { $sort: { total_gaps: -1 } },
    { $limit: 5 },
  ]);

  return analysis;
}

/**
 * Helper function to fetch spec frequency
 */
async function fetchSpecFrequency() {
  const frequency = await SkuGap.aggregate([
    { $unwind: "$missing_specs" },
    {
      $group: {
        _id: "$missing_specs.spec_name",
        count: { $sum: 1 },
      },
    },
    {
      $project: {
        _id: 0,
        spec_name: "$_id",
        count: 1,
      },
    },
    { $sort: { count: -1 } },
    { $limit: 10 },
  ]);

  return frequency;
}

/**
 * GET /api/sku-gap/insights
 * Generate AI-powered insights using Groq
 */
export const getAIInsights = async (req, res) => {
  try {
    // Step 1: Fetch analytics data
    const [summary, domainAnalysis, specFrequency] = await Promise.all([
      fetchSummary(),
      fetchDomainAnalysis(),
      fetchSpecFrequency(),
    ]);

    // Step 2: Build prompt for Groq
    const prompt = `You are a business intelligence analyst specializing in SKU gap analysis.

Based on the following SKU gap statistics:

SUMMARY:
${JSON.stringify(summary, null, 2)}

DOMAIN ANALYSIS (Top 5):
${JSON.stringify(domainAnalysis, null, 2)}

MOST MISSING SPECIFICATIONS (Top 10):
${JSON.stringify(specFrequency, null, 2)}

Please provide:
- 3 key insights about the current SKU coverage
- 3 specific SKU recommendations to address gaps
- 1 risk statement highlighting potential business impact
- 1 opportunity statement for market expansion

Return your analysis in concise bullet points, well-structured and actionable.`;

    // Step 3: Call Groq API
    const groqApiKey = process.env.GROQ_API_KEY;

    if (!groqApiKey) {
      return res.status(500).json({
        error: "Configuration error",
        message: "GROQ_API_KEY not found in environment variables",
      });
    }

    const response = await axios.post(
      "https://api.groq.com/openai/v1/chat/completions",
      {
        model: "llama-3.1-8b-instant",
        messages: [
          {
            role: "user",
            content: prompt,
          },
        ],
        temperature: 0.7,
        max_tokens: 1000,
      },
      {
        headers: {
          Authorization: `Bearer ${groqApiKey}`,
          "Content-Type": "application/json",
        },
        timeout: 15000,
      }
    );

    // Step 4: Return insights
    const insightsText = response.data.choices[0]?.message?.content || "No insights generated";

    res.json({
      insights_text: insightsText,
      data_summary: {
        total_items: summary.total_items,
        avg_match: summary.avg_match_percent,
        total_gaps: summary.total_missing_specs,
      },
    });
  } catch (error) {
    console.error("Error in getAIInsights:", error);
    
    // Log detailed error for debugging
    if (error.response) {
      console.error("Groq API Error Response:", error.response.data);
      console.error("Status:", error.response.status);
    }

    // Graceful fallback
    if (error.code === "ECONNABORTED" || error.response?.status === 429) {
      return res.status(503).json({
        error: "AI service temporarily unavailable",
        message: "Please try again in a moment",
        fallback: "Manual analysis recommended based on dashboard metrics",
      });
    }

    // Return more detailed error information
    const errorMessage = error.response?.data?.error?.message || error.message;
    
    res.status(500).json({
      error: "Failed to generate AI insights",
      message: errorMessage,
      fallback: "Review the analytics endpoints for detailed metrics",
    });
  }
};
