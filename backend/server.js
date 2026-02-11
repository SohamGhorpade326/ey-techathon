import express from "express";
import mongoose from "mongoose";
import cors from "cors";
import dotenv from "dotenv";

import productRoutes from "./routes/product.js";
import rfpRoutes from "./routes/rfps.js";
import testingMatrixRoutes from "./routes/testingMatrix.js";
import skuGapRoutes from "./routes/skuGap.routes.js";
import skuGapAIRoutes from "./routes/skuGapAI.routes.js";
import authRoutes from "./routes/auth.routes.js";
import { seedAdmin } from "./utils/seedAdmin.js";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

mongoose
  .connect(process.env.MONGO_URI)
  .then(() => {
    console.log("✅ MongoDB Connected");
    seedAdmin();
  })
  .catch((err) => console.error("Mongo Error:", err));

app.use("/api/auth", authRoutes);
app.use("/api/products", productRoutes);
app.use("/api/rfps", rfpRoutes);
app.use("/api/testing-matrix", testingMatrixRoutes);
app.use("/api/sku-gap", skuGapRoutes);
app.use("/api/sku-gap/insights", skuGapAIRoutes);

const PORT = process.env.PORT || 5000;
app.listen(PORT, () =>
  console.log(`🚀 Backend running on http://localhost:${PORT}`)
);
