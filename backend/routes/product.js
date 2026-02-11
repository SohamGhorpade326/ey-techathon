import express from "express";
import Product from "../models/Product.js";
import { verifyToken } from "../middleware/auth.middleware.js";
import { authorizeRoles } from "../middleware/role.middleware.js";

const router = express.Router();

// Apply auth middleware to all routes
router.use(verifyToken);
router.use(authorizeRoles("super_admin", "product"));

// Get all products
router.get("/", async (req, res) => {
  try {
    const products = await Product.find();
    res.json(products);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Create new product (dynamic fields supported)
router.post("/", async (req, res) => {
  try {
    console.log("Incoming payload:", req.body);
    const product = new Product(req.body);   // accepts dynamic fields
    await product.save();
    res.status(201).json(product);
  } catch (err) {
    res.status(400).json({ error: err.message });
  }
});
// Update product
router.put("/:id", async (req, res) => {
  try {
    const updated = await Product.findByIdAndUpdate(
      req.params.id,
      req.body,
      { new: true }
    );
    res.json(updated);
  } catch (err) {
    res.status(400).json({ error: err.message });
  }
});

// Delete product
router.delete("/:id", async (req, res) => {
  try {
    await Product.findByIdAndDelete(req.params.id);
    res.json({ success: true });
  } catch (err) {
    res.status(400).json({ error: err.message });
  }
});


export default router;

