/** @type {import('next').NextConfig} */
const nextConfig = {
  // Every page is pre-rendered, so the site ships as plain files to Firebase Hosting
  // (`npm run deploy`). `next build` writes them to out/.
  output: "export",
  images: { unoptimized: true },
  typescript: {
    ignoreBuildErrors: true,
  },
}

export default nextConfig
