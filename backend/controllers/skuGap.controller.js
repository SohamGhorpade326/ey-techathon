import SkuGap from "../models/SkuGap.js";

/**
 * GET /api/sku-gap/summary
 * Returns overall summary statistics
 */
export const getSummary = async (req, res) => {
  try {
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

    res.json({
      total_items: result.totalItems[0]?.count || 0,
      avg_match_percent: Math.round((result.avgMatch[0]?.avg || 0) * 100) / 100,
      full_count: result.fullCount[0]?.count || 0,
      partial_count: result.partialCount[0]?.count || 0,
      incomplete_count: result.incompleteCount[0]?.count || 0,
      total_missing_specs: result.totalMissingSpecs[0]?.total || 0,
    });
  } catch (error) {
    console.error("Error in getSummary:", error);
    res.status(500).json({
      error: "Failed to fetch summary",
      message: error.message,
    });
  }
};

/**
 * GET /api/sku-gap/domain-analysis
 * Returns gap analysis by domain
 */
export const getDomainAnalysis = async (req, res) => {
  try {
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
    ]);

    res.json(analysis);
  } catch (error) {
    console.error("Error in getDomainAnalysis:", error);
    res.status(500).json({
      error: "Failed to fetch domain analysis",
      message: error.message,
    });
  }
};

/**
 * GET /api/sku-gap/spec-frequency
 * Returns most frequently missing specifications
 */
export const getSpecFrequency = async (req, res) => {
  try {
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
    ]);

    res.json(frequency);
  } catch (error) {
    console.error("Error in getSpecFrequency:", error);
    res.status(500).json({
      error: "Failed to fetch spec frequency",
      message: error.message,
    });
  }
};

/**
 * GET /api/sku-gap/match-distribution
 * Returns distribution of match percentages
 */
export const getMatchDistribution = async (req, res) => {
  try {
    const distribution = await SkuGap.aggregate([
      {
        $facet: {
          high: [
            { $match: { spec_match_percent: { $gte: 80 } } },
            { $count: "count" },
          ],
          medium: [
            {
              $match: {
                spec_match_percent: { $gte: 60, $lt: 80 },
              },
            },
            { $count: "count" },
          ],
          low: [
            { $match: { spec_match_percent: { $lt: 60 } } },
            { $count: "count" },
          ],
        },
      },
    ]);

    const result = distribution[0];

    res.json({
      high: result.high[0]?.count || 0,
      medium: result.medium[0]?.count || 0,
      low: result.low[0]?.count || 0,
    });
  } catch (error) {
    console.error("Error in getMatchDistribution:", error);
    res.status(500).json({
      error: "Failed to fetch match distribution",
      message: error.message,
    });
  }
};

/**
 * GET /api/sku-gap/recommendations
 * Returns top recommendations based on missing specs
 */
export const getRecommendations = async (req, res) => {
  try {
    const recommendations = await SkuGap.aggregate([
      { $unwind: "$missing_specs" },
      {
        $group: {
          _id: "$missing_specs.spec_name",
          frequency: { $sum: 1 },
        },
      },
      { $sort: { frequency: -1 } },
      { $limit: 5 },
      {
        $project: {
          _id: 0,
          spec_name: "$_id",
          frequency: 1,
          suggestion: {
            $literal: "Consider adding SKU variant covering this specification",
          },
          impact_level: {
            $cond: {
              if: { $gt: ["$frequency", 15] },
              then: "High",
              else: {
                $cond: {
                  if: { $gt: ["$frequency", 8] },
                  then: "Medium",
                  else: "Low",
                },
              },
            },
          },
        },
      },
    ]);

    res.json(recommendations);
  } catch (error) {
    console.error("Error in getRecommendations:", error);
    res.status(500).json({
      error: "Failed to fetch recommendations",
      message: error.message,
    });
  }
};

/**
 * GET /api/sku-gap/timeline
 * Returns timeline analysis of match percentages
 */
export const getTimeline = async (req, res) => {
  try {
    const timeline = await SkuGap.aggregate([
      {
        $group: {
          _id: {
            $dateToString: {
              format: "%Y-%m-%d",
              date: "$created_at",
            },
          },
          avg_match: { $avg: "$spec_match_percent" },
        },
      },
      {
        $project: {
          _id: 0,
          date: "$_id",
          avg_match: { $round: ["$avg_match", 2] },
        },
      },
      { $sort: { date: 1 } },
    ]);

    res.json(timeline);
  } catch (error) {
    console.error("Error in getTimeline:", error);
    res.status(500).json({
      error: "Failed to fetch timeline",
      message: error.message,
    });
  }
};

/**
 * GET /api/sku-gap/risk-value
 * Returns risk assessment and estimated value
 */
export const getRiskValue = async (req, res) => {
  try {
    const riskAnalysis = await SkuGap.aggregate([
      {
        $facet: {
          riskItems: [
            { $match: { spec_match_percent: { $lt: 70 } } },
            { $count: "count" },
          ],
        },
      },
    ]);

    const riskCount = riskAnalysis[0].riskItems[0]?.count || 0;

    res.json({
      risk_items: riskCount,
      estimated_risk_score: riskCount * 1000000,
    });
  } catch (error) {
    console.error("Error in getRiskValue:", error);
    res.status(500).json({
      error: "Failed to fetch risk value",
      message: error.message,
    });
  }
};
