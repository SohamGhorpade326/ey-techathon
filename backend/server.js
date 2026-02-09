import express from "express";
import mongoose from "mongoose";
import cors from "cors";
import dotenv from "dotenv";

import productRoutes from "./routes/product.js";
import rfpRoutes from "./routes/rfps.js";
import testingMatrixRoutes from "./routes/testingMatrix.js";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

mongoose
  .connect(process.env.MONGO_URI)
  .then(() => console.log("✅ MongoDB Connected"))
  .catch((err) => console.error("Mongo Error:", err));

app.use("/api/products", productRoutes);
app.use("/api/rfps", rfpRoutes);
app.use("/api/testing-matrix", testingMatrixRoutes);

const PORT = process.env.PORT || 5000;
app.listen(PORT, () =>
  console.log(`🚀 Backend running on http://localhost:${PORT}`)
);
