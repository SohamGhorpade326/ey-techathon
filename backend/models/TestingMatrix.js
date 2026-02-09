import mongoose from "mongoose";

const testingMatrixSchema = new mongoose.Schema(
  {
    test_code: String,
    test_name: String,
    applicable_product_types: [String],
    standard_reference: String,
    test_category: String,
    price_inr: Number,
    price_basis: String
  },
  {
    collection: "test_data" // 🔥 THIS IS THE FIX
  }
);

export default mongoose.model("TestingMatrix", testingMatrixSchema);
