import fs from "fs";
import path from "path";
import process from "process";
import readline from "readline";
import crypto from "crypto";

import dotenv from "dotenv";
import { google } from "googleapis";
import { PubSub } from "@google-cloud/pubsub";

dotenv.config();

// ---------------------------------------------------------------------------
// STRICT RULES
// - Do NOT modify or print credentials.json.
// - Gmail logic stays inside gmail-rfp-alert/.
// - Fail silently if Gmail API is temporarily unavailable (no crash).
// ---------------------------------------------------------------------------

const KEYWORDS = ["rfp", "proposal", "tender", "bid invitation", "bid reference", "submission deadline"];

const BACKEND_URL = process.env.BACKEND_URL || "http://127.0.0.1:8000";
const INGEST_ENDPOINT = `${BACKEND_URL}/integrations/gmail/rfp-event`;

const GMAIL_USER = process.env.GMAIL_USER || "me";

// 10-minute time window for recent RFP emails
const TEN_MINUTES_MS = 10 * 60 * 1000;

function isWithinTenMinutes(emailInternalDate) {
  const now = Date.now();
  const emailTime = Number(emailInternalDate); // Gmail internalDate is epoch milliseconds
  return (now - emailTime) <= TEN_MINUTES_MS;
}

// Pub/Sub identifiers used by Gmail push notifications
const GCP_PROJECT_ID = process.env.GCP_PROJECT_ID;
const PUBSUB_SUBSCRIPTION = process.env.PUBSUB_SUBSCRIPTION; // e.g. projects/<project>/subscriptions/<sub>
const PUBSUB_TOPIC = process.env.PUBSUB_TOPIC; // e.g. projects/<project>/topics/<topic>

const CREDENTIALS_PATH = path.resolve("./credentials.json");
const TOKEN_PATH = path.resolve("./token.json");
const STATE_PATH = path.resolve("./.state.json");

const DEBUG = process.env.DEBUG === "1";

function debugLog(...args) {
  if (DEBUG) console.log("[gmail-rfp-alert]", ...args);
}

function safeJsonRead(filePath, fallback) {
  try {
    if (!fs.existsSync(filePath)) return fallback;
    return JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch {
    return fallback;
  }
}

function safeJsonWrite(filePath, value) {
  try {
    fs.writeFileSync(filePath, JSON.stringify(value, null, 2), "utf8");
  } catch {
    // Fail silently
  }
}

function looksLikeRfpSubject(subject) {
  const s = (subject || "").toLowerCase();
  return KEYWORDS.some((k) => s.includes(k));
}

function looksLikeRfpContent(subject, bodyText) {
  const s = (subject || "").toLowerCase();
  const b = (bodyText || "").toLowerCase();
  return KEYWORDS.some((k) => s.includes(k) || b.includes(k));
}

function isTenderEmail(subject, bodyText) {
  const s = (subject || "").toLowerCase();
  const b = (bodyText || "").toLowerCase();

  // Rule 1: subject contains at least one tender/RFP keyword.
  const subjectKeywords = [
    "rfp",
    "tender",
    "request for proposal",
    "bid reference",
    "quotation invited",
  ];
  if (subjectKeywords.some((k) => s.includes(k))) return true;

  // Rule 2: body contains BOTH a deadline pattern AND an estimated value pattern.
  const deadlineRegex =
    /(submission\s*deadline|bid\s*submission|last\s*date|closing\s*date|due\s*date|deadline)\b|\b\d{1,2}\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s*\d{4}\b|\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b/;
  const valueRegex =
    /(estimated\s*value|tender\s*value)\b|\b(inr|rs\.?|rupees)\b|₹|\b(crore|cr)\b|\b(lakh|lakhs)\b/;

  return deadlineRegex.test(b) && valueRegex.test(b);
}

function decodeBase64Url(b64url) {
  if (!b64url) return "";
  try {
    const b64 = String(b64url).replace(/-/g, "+").replace(/_/g, "/");
    const pad = b64.length % 4;
    const padded = pad ? b64 + "=".repeat(4 - pad) : b64;
    return Buffer.from(padded, "base64").toString("utf8");
  } catch {
    return "";
  }
}

function extractBodyText(payload) {
  try {
    if (!payload) return "";
    const parts = [];

    const walk = (node) => {
      if (!node) return;
      const mime = (node.mimeType || "").toLowerCase();
      const data = node.body?.data;
      if (data && (mime.startsWith("text/plain") || mime === "")) {
        const txt = decodeBase64Url(data);
        if (txt) parts.push(txt);
      }
      for (const p of node.parts || []) walk(p);
    };

    walk(payload);
    const joined = parts.join("\n").trim();
    return joined;
  } catch {
    return "";
  }
}

function extractEmail(fromHeader) {
  if (!fromHeader) return "";
  const match = String(fromHeader).match(/<([^>]+)>/);
  return match ? match[1] : String(fromHeader);
}

function createStableEmailId(sender, subject, date) {
  // Create a content-based ID to deduplicate emails sent to self
  // (Gmail creates separate message IDs for Sent and Inbox)
  const basis = `${sender}|${subject}|${date}`;
  return crypto.createHash('md5').update(basis).digest('hex').substring(0, 16);
}

async function postToBackend(event) {
  try {
    const res = await fetch(INGEST_ENDPOINT, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(event),
    });
    // If backend is down, fail silently.
    if (!res.ok) {
      debugLog("Backend returned non-OK:", res.status);
      return;
    }
  } catch {
    // Fail silently.
    debugLog("Backend unreachable; event dropped");
  }
}

async function authorize() {
  // NOTE: We must not print credentials.json contents.
  const raw = JSON.parse(fs.readFileSync(CREDENTIALS_PATH, "utf8"));
  const creds = raw.installed || raw.web;
  if (!creds) throw new Error("Invalid credentials.json format");

  const oAuth2Client = new google.auth.OAuth2(
    creds.client_id,
    creds.client_secret,
    creds.redirect_uris?.[0]
  );

  const token = safeJsonRead(TOKEN_PATH, null);
  if (token) {
    oAuth2Client.setCredentials(token);
    return oAuth2Client;
  }

  // First-time auth (interactive). OAuth setup already done on GCP.
  const authUrl = oAuth2Client.generateAuthUrl({
    access_type: "offline",
    scope: ["https://www.googleapis.com/auth/gmail.readonly"],
    prompt: "consent",
  });

  // Minimal prompt; do not dump secrets.
  console.log("Authorize this app by visiting this url:\n", authUrl);
  console.log(
    "If the browser redirects to http://localhost and shows 'refused to connect',\n" +
      "that is OK. Copy the full URL from the address bar (it contains ?code=...)\n" +
      "and paste it here."
  );

  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  const codeOrUrl = await new Promise((resolve) =>
    rl.question("Paste the code (or the full redirected URL) here: ", resolve)
  );
  rl.close();

  // Accept either a raw code or a full URL like: http://localhost/?code=...&scope=...
  let code = String(codeOrUrl || "").trim();
  try {
    if (code.startsWith("http://") || code.startsWith("https://")) {
      const u = new URL(code);
      const c = u.searchParams.get("code");
      if (c) code = c;
    }
  } catch {
    // ignore
  }

  const { tokens } = await oAuth2Client.getToken(code);
  oAuth2Client.setCredentials(tokens);
  safeJsonWrite(TOKEN_PATH, tokens);
  return oAuth2Client;
}

async function ensureWatch(gmail, state) {
  if (!PUBSUB_TOPIC) {
    throw new Error("PUBSUB_TOPIC env var is required for Gmail watch");
  }

  // Refresh the watch to keep it alive.
  const res = await gmail.users.watch({
    userId: GMAIL_USER,
    requestBody: {
      topicName: PUBSUB_TOPIC,
      labelIds: ["INBOX"],
    },
  });

  const historyId = res.data.historyId;
  if (historyId) {
    state.lastHistoryId = String(historyId);
    safeJsonWrite(STATE_PATH, state);
    debugLog("Watch refreshed. lastHistoryId=", state.lastHistoryId);
  }
}

async function bootstrapRecentMessages(gmail) {
  try {
    const list = await gmail.users.messages.list({
      userId: GMAIL_USER,
      labelIds: ["INBOX"],
      q: "is:unread (rfp OR proposal OR tender OR bid)",
      maxResults: 10,
    });

    const ids = (list.data.messages || []).map((m) => m.id).filter(Boolean);
    for (const id of ids) {
      try {
        const msg = await gmail.users.messages.get({
          userId: GMAIL_USER,
          id,
          format: "full",
        });

        const headers = msg.data.payload?.headers || [];
        const subject = headers.find((h) => h.name === "Subject")?.value || "";
        const from = headers.find((h) => h.name === "From")?.value || "";
        const date = headers.find((h) => h.name === "Date")?.value || "";

        const body_text = extractBodyText(msg.data.payload);
        if (!isTenderEmail(subject, body_text)) continue;

        const sender = extractEmail(from);
        const timestamp = date ? new Date(date).toISOString() : new Date().toISOString();

        debugLog("Bootstrap match:", { subject, sender, timestamp });
        await postToBackend({
          message_id: id,
          sender,
          subject,
          body_text,
          received_timestamp: timestamp,
        });
      } catch {
        // silent
      }
    }
  } catch {
    // silent
  }
}

async function processHistory(gmail, state, historyId, processedEmails) {
  // DISABLED: Bootstrap was causing duplicate notifications from old emails
  // Only process NEW messages going forward
  if (!state.lastHistoryId) {
    // Just set the checkpoint without fetching old emails
    state.lastHistoryId = String(historyId);
    safeJsonWrite(STATE_PATH, state);
    debugLog("First run: Set checkpoint without bootstrapping old emails");
    return;
  }

  const startHistoryId = state.lastHistoryId;

  let pageToken = undefined;
  const messageIds = new Set(); // Use Set to automatically deduplicate message IDs

  while (true) {
    const resp = await gmail.users.history.list({
      userId: GMAIL_USER,
      startHistoryId,
      historyTypes: ["messageAdded"],
      pageToken,
    });

    const history = resp.data.history || [];
    for (const h of history) {
      const added = h.messagesAdded || [];
      for (const m of added) {
        if (m.message?.id) messageIds.add(m.message.id); // Use .add() for Set
      }
    }

    pageToken = resp.data.nextPageToken;
    if (!pageToken) break;
  }

  // Advance lastHistoryId to the newest we saw.
  state.lastHistoryId = String(historyId);
  safeJsonWrite(STATE_PATH, state);

  // Convert Set to Array for iteration
  for (const id of Array.from(messageIds)) {
    try {
      const msg = await gmail.users.messages.get({
        userId: GMAIL_USER,
        id,
        format: "full",
      });

      // FIX 1: ONLY process INBOX messages (skip SENT)
      const labelIds = msg.data.labelIds || [];
      if (!labelIds.includes("INBOX")) {
        debugLog("Skipping non-INBOX message:", id);
        continue;
      }

      // FIX 2: 10-MINUTE TIME WINDOW - Only process recent emails
      const internalDate = msg.data.internalDate; // Gmail epoch milliseconds
      if (!isWithinTenMinutes(internalDate)) {
        debugLog("Skipping old message (older than 10 minutes):", id);
        continue;
      }

      // TESTING MODE: Deduplication temporarily disabled for multiple test emails
      const messageId = msg.data.id;
      // if (processedEmails.has(messageId)) {
      //   debugLog("Skipping duplicate messageId:", messageId);
      //   continue;
      // }
      // processedEmails.add(messageId);

      // FIX 3: Extract subject correctly from headers
      const headers = msg.data.payload?.headers || [];
      const headersMap = {};
      headers.forEach(h => {
        if (h.name) headersMap[h.name] = h.value || "";
      });
      const subject = headersMap["Subject"] || "No Subject";
      const from = headersMap["From"] || "";
      const date = headersMap["Date"] || "";

      // FIX 4: Store FULL body (decoded)
      const body_text = extractBodyText(msg.data.payload);
      if (!isTenderEmail(subject, body_text)) continue;

      const sender = extractEmail(from);
      const timestamp = date ? new Date(date).toISOString() : new Date(Number(internalDate)).toISOString();
      const snippet = msg.data.snippet || "";

      debugLog("Recent RFP match (within 10 min):", { messageId, subject, sender, timestamp });
      
      // Send full data including messageId, snippet, and complete body
      await postToBackend({
        message_id: messageId,
        sender,
        subject,
        body_text,
        snippet,
        received_timestamp: timestamp,
      });
    } catch (e) {
      // Log errors in debug mode
      debugLog("Error processing message:", e?.message || e);
    }
  }
}

async function startSubscriber(gmail) {
  if (!GCP_PROJECT_ID || !PUBSUB_SUBSCRIPTION) {
    throw new Error("GCP_PROJECT_ID and PUBSUB_SUBSCRIPTION env vars are required");
  }

  const pubsub = new PubSub({ projectId: GCP_PROJECT_ID });
  const subscriptionName = PUBSUB_SUBSCRIPTION.split("/subscriptions/")[1] || PUBSUB_SUBSCRIPTION;
  const subscription = pubsub.subscription(subscriptionName);

  // Debug-only: verify we can access the subscription (catches missing ADC / permissions).
  try {
    await subscription.getMetadata();
    debugLog("Pub/Sub subscription OK:", subscriptionName);
  } catch (e) {
    debugLog(
      "Pub/Sub subscription metadata failed (likely auth/permissions). " +
        "If you haven't set Google ADC, run: gcloud auth application-default login",
      e?.message || e
    );
  }

  const state = safeJsonRead(STATE_PATH, { lastHistoryId: null });

  // Track processed historyIds to prevent duplicate processing from Pub/Sub redelivery
  const processedHistoryIds = new Set();
  const MAX_HISTORY_CACHE = 1000; // Limit memory usage
  
  // Track processed emails by content to prevent duplicate notifications for self-sent emails
  const processedEmails = new Set();
  const MAX_EMAIL_CACHE = 500; // Limit memory usage

  // Optional: keep watch alive periodically (Gmail watch expires).
  const shouldRenewWatch = process.argv.includes("--watch");
  if (shouldRenewWatch) {
    setInterval(async () => {
      try {
        await ensureWatch(gmail, state);
      } catch {
        // Fail silently.
      }
    }, 1000 * 60 * 30); // every 30 min
  }

  subscription.on("message", async (message) => {
    try {
      const payload = JSON.parse(message.data.toString("utf8"));
      const historyId = payload.historyId;
      if (!historyId) {
        message.ack();
        return;
      }

      const historyIdStr = String(historyId);
      
      // Deduplicate: skip if we've already processed this historyId
      if (processedHistoryIds.has(historyIdStr)) {
        debugLog("Skipping duplicate historyId=", historyIdStr);
        message.ack();
        return;
      }

      debugLog("Pub/Sub event historyId=", historyIdStr);
      
      // Mark as processed before processing to handle concurrent deliveries
      processedHistoryIds.add(historyIdStr);
      
      // Memory management: clear oldest entries if cache grows too large
      if (processedHistoryIds.size > MAX_HISTORY_CACHE) {
        const toDelete = Array.from(processedHistoryIds).slice(0, MAX_HISTORY_CACHE / 2);
        toDelete.forEach(id => processedHistoryIds.delete(id));
        debugLog(`Cleared ${toDelete.length} old historyIds from cache`);
      }
      
      // Memory management for email cache
      if (processedEmails.size > MAX_EMAIL_CACHE) {
        const toDelete = Array.from(processedEmails).slice(0, MAX_EMAIL_CACHE / 2);
        toDelete.forEach(id => processedEmails.delete(id));
        debugLog(`Cleared ${toDelete.length} old email IDs from cache`);
      }

      await processHistory(gmail, state, historyIdStr, processedEmails);
      message.ack();
    } catch (e) {
      // Log errors in debug mode, ack to avoid retries that may spike latency
      debugLog("Error in subscription handler:", e?.message || e);
      try {
        message.ack();
      } catch {
        // ignore
      }
    }
  });

  subscription.on("error", (e) => {
    // Log error in debug mode
    debugLog("Pub/Sub subscription error:", e?.message || e);
  });
}

async function main() {
  try {
    const auth = await authorize();
    const gmail = google.gmail({ version: "v1", auth });

    const state = safeJsonRead(STATE_PATH, { lastHistoryId: null });

    // One-time watch call (optional if already set up). If it fails, we continue.
    try {
      await ensureWatch(gmail, state);
    } catch {
      // Fail silently.
      debugLog("Gmail watch setup failed (topic/permissions?)");
    }

    await startSubscriber(gmail);
    console.log("gmail-rfp-alert listener is running");
  } catch {
    // Fail silently if Gmail is temporarily unavailable.
    // Keep process alive so it can recover after environment fixes.
    setInterval(() => {}, 1 << 30);
  }
}

main();
