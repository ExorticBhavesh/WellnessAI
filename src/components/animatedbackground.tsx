import { motion } from "framer-motion";

export default function AnimatedBackground() {
  return (
    <div className="fixed inset-0 -z-10 overflow-hidden">
      {/* Gradient base */}
      <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/10 via-cyan-400/10 to-blue-500/10" />

      {/* Floating blobs */}
      <motion.div
        className="absolute -top-32 -left-32 h-96 w-96 rounded-full bg-emerald-400/20 blur-3xl"
        animate={{ y: [0, 40, 0], x: [0, 30, 0] }}
        transition={{ duration: 18, repeat: Infinity, ease: "easeInOut" }}
      />

      <motion.div
        className="absolute top-1/3 -right-32 h-96 w-96 rounded-full bg-cyan-400/20 blur-3xl"
        animate={{ y: [0, -50, 0], x: [0, -40, 0] }}
        transition={{ duration: 22, repeat: Infinity, ease: "easeInOut" }}
      />

      <motion.div
        className="absolute bottom-0 left-1/4 h-96 w-96 rounded-full bg-blue-500/20 blur-3xl"
        animate={{ y: [0, 60, 0] }}
        transition={{ duration: 26, repeat: Infinity, ease: "easeInOut" }}
      />
    </div>
  );
}
