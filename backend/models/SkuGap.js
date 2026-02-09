import mongoose from "mongoose";

const skuGapSchema = new mongoose.Schema(
  {
    item_name: {
      type: String,
      required: true,
    },

    rfp_id: {
      type: String,
      required: true,
    },

    best_match_sku: {
      type: String,
    },

    category: {
      type: String,
    },

    domain: {
      type: String,
      index: true,
    },

    occurrences: {
      type: Number,
      default: 0,
    },

    spec_match_percent: {
      type: Number,
      min: 0,
      max: 100,
      index: true,
    },

    // IMPORTANT → objects, not strings
    missing_specs: [
      {
        spec_name: String,
        rfp_value: mongoose.Schema.Types.Mixed,
        sku_value: mongoose.Schema.Types.Mixed,
      },
    ],

    created_at: {
      type: Date,
      default: Date.now,
      index: true,
    },

    last_seen: {
      type: Date,
      default: Date.now,
    },
  },
  {
    timestamps: false, // we already manage dates manually
  }
);

/*
🔥 MOST IMPORTANT FIX
Must EXACTLY match Mongo collection name
*/
export default mongoose.model(
  "SkuGap",
  skuGapSchema,
  "sku_gap_intelligence"
);
