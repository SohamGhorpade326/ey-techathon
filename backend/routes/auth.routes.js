import express from "express";
import { login, createUser } from "../controllers/auth.controller.js";
import { verifyToken } from "../middleware/auth.middleware.js";
import { authorizeRoles } from "../middleware/role.middleware.js";

const router = express.Router();

// Public route
router.post("/login", login);

// Protected route - only super_admin can create users
router.post("/create-user", verifyToken, authorizeRoles("super_admin"), createUser);

export default router;
