import express from "express";
import Rfp from "../models/RFP.js";
import { verifyToken } from "../middleware/auth.middleware.js";
import { authorizeRoles } from "../middleware/role.middleware.js";

const router = express.Router();

// Apply auth middleware to all routes
router.use(verifyToken);
router.use(authorizeRoles("super_admin", "sales", "technical"));

// Get all RFPs
router.get("/", async (req, res) => {
  const rfps = await Rfp.find();
  res.json(rfps);
});

// Get single RFP by ID
router.get("/:id", async (req, res) => {
  const rfp = await Rfp.findOne({ rfp_id: req.params.id });
  res.json(rfp);
});

export default router;
