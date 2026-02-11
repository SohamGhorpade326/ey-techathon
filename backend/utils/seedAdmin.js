import User from "../models/User.js";

export const seedAdmin = async () => {
  try {
    const adminEmail = "admin@agenticrfp.com";
    const existingAdmin = await User.findOne({ email: adminEmail });

    if (existingAdmin) {
      console.log("✅ Default admin already exists");
      return;
    }

    const admin = new User({
      name: "Super Admin",
      email: adminEmail,
      password: "Admin@123",
      role: "super_admin",
      isActive: true,
    });

    await admin.save();

    console.log("\n" + "=".repeat(60));
    console.log("🔐 DEFAULT ADMIN ACCOUNT CREATED");
    console.log("=".repeat(60));
    console.log(`📧 Email:    ${adminEmail}`);
    console.log(`🔑 Password: Admin@123`);
    console.log(`👤 Role:     super_admin`);
    console.log("=".repeat(60) + "\n");
  } catch (error) {
    console.error("❌ Error seeding admin:", error);
  }
};
