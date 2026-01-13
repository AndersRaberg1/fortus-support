/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    serverComponentsExternalPackages: ['google-spreadsheet'],
  },
};

module.exports = nextConfig;
