import express from "express";
import * as skuGapController from "../controllers/skuGap.controller.js";
import { verifyToken } from "../middleware/auth.middleware.js";
import { authorizeRoles } from "../middleware/role.middleware.js";

const router = express.Router();

// Apply auth middleware to all routes
router.use(verifyToken);
router.use(authorizeRoles("super_admin", "product", "technical", "pricing"));

/**
 * @route   GET /api/sku-gap/summary
 * @desc    Get overall SKU gap summary statistics
 * @access  Public
 */
router.get("/summary", skuGapController.getSummary);

/**
 * @route   GET /api/sku-gap/domain-analysis
 * @desc    Get gap analysis by domain
 * @access  Public
 */
router.get("/domain-analysis", skuGapController.getDomainAnalysis);

/**
 * @route   GET /api/sku-gap/spec-frequency
 * @desc    Get most frequently missing specifications
 * @access  Public
 */
router.get("/spec-frequency", skuGapController.getSpecFrequency);

/**
 * @route   GET /api/sku-gap/match-distribution
 * @desc    Get distribution of match percentages
 * @access  Public
 */
router.get("/match-distribution", skuGapController.getMatchDistribution);

/**
 * @route   GET /api/sku-gap/recommendations
 * @desc    Get top recommendations based on missing specs
 * @access  Public
 */
router.get("/recommendations", skuGapController.getRecommendations);

/**
 * @route   GET /api/sku-gap/timeline
 * @desc    Get timeline analysis of match percentages
 * @access  Public
 */
router.get("/timeline", skuGapController.getTimeline);

/**
 * @route   GET /api/sku-gap/risk-value
 * @desc    Get risk assessment and estimated value
 * @access  Public
 */
router.get("/risk-value", skuGapController.getRiskValue);

export default router;
