import express from "express";
import * as skuGapAIController from "../controllers/skuGapAI.controller.js";
import { verifyToken } from "../middleware/auth.middleware.js";
import { authorizeRoles } from "../middleware/role.middleware.js";

const router = express.Router();

// Apply auth middleware to all routes
router.use(verifyToken);
router.use(authorizeRoles("super_admin", "product", "technical", "pricing"));

/**
 * @route   GET /api/sku-gap/insights
 * @desc    Generate AI-powered insights using Groq
 * @access  Public
 */
router.get("/", skuGapAIController.getAIInsights);

export default router;
