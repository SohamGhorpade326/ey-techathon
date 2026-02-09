import mongoose from "mongoose";

const productSchema = new mongoose.Schema(
  {},
  {
    strict: false,     // ✅ allows dynamic fields
    timestamps: true
  }
);

export default mongoose.model("Product", productSchema, "product_catalog");
