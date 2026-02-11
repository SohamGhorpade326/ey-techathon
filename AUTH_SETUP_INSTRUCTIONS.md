# Authentication & RBAC Implementation - Setup Instructions

## ⚠️ IMPORTANT: Install Required Dependencies

### Backend Dependencies
Navigate to the backend folder and install:

```bash
cd backend
npm install bcryptjs jsonwebtoken
```

These packages are required for:
- `bcryptjs` - Password hashing
- `jsonwebtoken` - JWT token generation and verification

### Environment Variables
Add to your `.env` file in the backend folder:

```
JWT_SECRET=your-super-secret-jwt-key-change-this-in-production
```

## 🚀 Starting the Application

### 1. Start Backend
```bash
cd backend
npm start
```

You will see the default admin credentials printed in the console:

```
============================================================
🔐 DEFAULT ADMIN ACCOUNT CREATED
============================================================
📧 Email:    admin@agenticrfp.com
🔑 Password: Admin@123
👤 Role:     super_admin
============================================================
```

### 2. Start Frontend
```bash
cd RFPilot
npm run dev
```

### 3. Login
- Navigate to `http://localhost:5173/login`
- Use the default admin credentials shown above
- You will be redirected to the dashboard upon successful login

## 🔐 Security Features Implemented

### Backend
✅ User model with password hashing (bcrypt)
✅ JWT-based authentication
✅ Role-based access control middleware
✅ Protected API routes with role authorization
✅ Admin-only user creation endpoint
✅ Default admin seeding on server start

### Frontend
✅ Login page connected to backend API
✅ JWT token storage in localStorage
✅ Role badge display in TopBar
✅ Logout functionality
✅ Protected routes with ProtectedRoute component
✅ Role-based route access control
✅ "Access Denied" page for unauthorized access

## 👥 Role Access Matrix

| Page/Route | Allowed Roles |
|------------|---------------|
| Dashboard | ALL roles |
| RFP Discovery | super_admin, sales, technical |
| Strategic Orchestrator (Master Agent) | super_admin, sales, technical, product |
| AI Analysis Workspace | super_admin, technical, product |
| Pricing & Summary | super_admin, pricing |
| Proposal Builder | super_admin, sales, pricing |
| SKU Gap Intelligence | super_admin, product, technical, pricing |
| Notifications | ALL roles |
| Product Repository | super_admin, product |
| Settings | super_admin ONLY |

## 🔑 Available Roles

- `super_admin` - Full access to all features
- `sales` - Sales-related features
- `technical` - Technical analysis and specifications
- `pricing` - Pricing and cost analysis
- `product` - Product repository and SKU management

## 📝 Creating New Users

Only super_admin can create new users via API:

```bash
POST http://localhost:5000/api/auth/create-user
Headers:
  Authorization: Bearer <super_admin_jwt_token>
  Content-Type: application/json

Body:
{
  "name": "John Doe",
  "email": "john@company.com",
  "password": "SecurePassword123",
  "role": "sales"
}
```

## 🧪 Testing

1. Login as admin
2. Try accessing different pages
3. Create users with different roles (via API)
4. Login with different role users
5. Verify access restrictions work correctly

## ⚡ What Was Changed

### New Backend Files
- `models/User.js` - User schema with roles
- `controllers/auth.controller.js` - Login and user creation
- `routes/auth.routes.js` - Auth endpoints
- `middleware/auth.middleware.js` - JWT verification
- `middleware/role.middleware.js` - Role authorization
- `utils/seedAdmin.js` - Default admin seeding

### Modified Backend Files
- `server.js` - Added auth routes and admin seeding
- All route files - Added auth middleware with role restrictions

### New Frontend Files
- `components/ProtectedRoute.tsx` - Route guard component

### Modified Frontend Files
- `App.tsx` - Wrapped routes with ProtectedRoute
- `pages/Login.tsx` - Connected to backend API
- `components/layout/TopBar.tsx` - Added role badge and logout

## ✨ NO Business Logic Changes
- All existing features work exactly the same
- No layouts or styling changed
- No analytics or pipelines modified
- Only security layer added on top
