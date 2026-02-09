import mongoose from "mongoose";

const rfpSchema = new mongoose.Schema({
  rfp_id: String,
  title: String,
  buyer: String,
  portal: String,
  deadline: String,
  estimated_value: String,
  description: String,
  scope_summary: [String],
  testing_requirements: [String],
  priority: String,
  status: String,
  days_remaining: Number
});

export default mongoose.model("Rfp", rfpSchema);
