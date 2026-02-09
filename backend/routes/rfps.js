import express from "express";
import Rfp from "../models/RFP.js";

const router = express.Router();

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
