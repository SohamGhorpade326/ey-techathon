import express from "express";
import TestingMatrix from "../models/TestingMatrix.js";

const router = express.Router();

// GET ALL
router.get("/", async (req, res) => {
  const data = await TestingMatrix.find();
  res.json(data);
});

// CREATE
router.post("/", async (req, res) => {
  const item = new TestingMatrix(req.body);
  await item.save();
  res.json(item);
});

// UPDATE
router.put("/:id", async (req, res) => {
  const updated = await TestingMatrix.findByIdAndUpdate(
    req.params.id,
    req.body,
    { new: true }
  );
  res.json(updated);
});

// DELETE
router.delete("/:id", async (req, res) => {
  await TestingMatrix.findByIdAndDelete(req.params.id);
  res.json({ success: true });
});

export default router;
