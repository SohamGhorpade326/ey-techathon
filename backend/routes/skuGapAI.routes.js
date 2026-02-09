import express from "express";
import * as skuGapAIController from "../controllers/skuGapAI.controller.js";

const router = express.Router();

/**
 * @route   GET /api/sku-gap/insights
 * @desc    Generate AI-powered insights using Groq
 * @access  Public
 */
router.get("/", skuGapAIController.getAIInsights);

export default router;
