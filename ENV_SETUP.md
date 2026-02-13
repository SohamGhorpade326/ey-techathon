# Environment Variables Setup Guide

This document lists all the environment variables needed across different parts of the project.

## 🔐 Security Note
**NEVER commit `.env` files to GitHub!** All `.env` files are in `.gitignore`. Use `.env.example` files as templates.

---

## 📁 RFPPilot2/backend/.env

Required environment variables for the main Python backend:

```env
# MongoDB Atlas Connection
MONGO_URI2=mongodb+srv://username:password@cluster.mongodb.net/?appName=Cluster0

# Server Configuration
PORT=5000

# Gmail SMTP Configuration (for sending bid emails)
GMAIL_SENDER_EMAIL=your-email@gmail.com
GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx

# Groq API Key (for AI cover letter generation)
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### How to Get These:
- **MongoDB**: Create cluster at [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
- **Gmail App Password**: 
  1. Go to [Google Account](https://myaccount.google.com/)
  2. Security → 2-Step Verification → App passwords
  3. Generate password for "Mail"
- **Groq API Key**: Get from [Groq Console](https://console.groq.com/keys)

---

## 📁 backend/.env

Required for the Node.js backend (user management, SKU gap analysis):

```env
# MongoDB URI
MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/database_name

# Server Port
PORT=5000

# Groq API Key for AI Insights
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# JWT Secret for Authentication
JWT_SECRET=your-super-secret-jwt-key-change-this-in-production
```

---

## 📁 RFPilot/.env

Required for the React frontend (Vite):

```env
# Backend API URL
VITE_API_URL=http://127.0.0.1:8000

# Groq API Key (for frontend AI features)
VITE_GROQ_API_KEY=your_groq_api_key_here
```

**Note**: Vite environment variables MUST start with `VITE_` to be exposed to the client.

---

## 📁 gmail-rfp-alert/

### credentials.json
Get from [Google Cloud Console](https://console.cloud.google.com/):
1. Create a project
2. Enable Gmail API
3. Create OAuth 2.0 credentials (Desktop app)
4. Download as `credentials.json`

### token.json
Generated automatically after first OAuth flow. Keep this secret!

---

## 🚀 Setup Instructions

### 1. Clone the repository
```bash
git clone <your-repo>
cd ey-techathon
```

### 2. Copy example files and fill in your credentials
```bash
# Backend Python
cp RFPPilot2/backend/.env.example RFPPilot2/backend/.env
# Edit RFPPilot2/backend/.env with your actual values

# Backend Node.js
cp backend/.env.example backend/.env
# Edit backend/.env with your actual values

# Frontend React
cp RFPilot/.env.example RFPilot/.env
# Edit RFPilot/.env with your actual values

# Gmail Listener
cp gmail-rfp-alert/credentials.json.example gmail-rfp-alert/credentials.json
# Download your actual credentials.json from Google Cloud Console
```

### 3. Install dependencies
```bash
# Python backend
cd RFPPilot2/backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Node.js backend
cd ../../backend
npm install

# React frontend
cd ../RFPilot
npm install

# Gmail listener
cd ../gmail-rfp-alert
npm install
```

### 4. Run the application
```bash
# Terminal 1: Python backend
cd RFPPilot2/backend
venv\Scripts\activate
python -m uvicorn main_new:app --reload

# Terminal 2: Node.js backend
cd backend
npm start

# Terminal 3: React frontend
cd RFPilot
npm run dev

# Terminal 4 (optional): Gmail listener
cd gmail-rfp-alert
node listener.js
```

---

## ⚠️ Important Notes

1. **Never share your `.env` files** - They contain sensitive credentials
2. **Regenerate all keys** after cloning if you're setting up a new instance
3. **Update `.gitignore`** if you add new sensitive files
4. **Use strong JWT_SECRET** - Generate with: `node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"`
5. **Gmail App Password** - Regular password won't work, must use App Password

---

## 📧 Email Sending Notes

The system uses **Gmail SMTP** (not SendGrid) for sending bid emails:
- Ensure 2-Step Verification is enabled on your Google Account
- Generate an App Password specifically for this application
- Don't use your regular Gmail password

---

## 🔍 Troubleshooting

**MongoDB Connection Failed**: Check your IP is whitelisted in MongoDB Atlas Network Access

**Gmail SMTP 535 Error**: Verify you're using App Password, not regular password

**Frontend API calls fail**: Ensure VITE_API_URL matches your backend URL

**Groq API 401**: Verify your API key is valid and has credits

---

## 📝 License
This is a Techathon project. Update license information as needed.
