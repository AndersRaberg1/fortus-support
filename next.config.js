/** @type {import('next').NextConfig} */
const nextConfig = {
  webpack: (config, { isServer }) => {
    if (isServer) {
      config.externals.push('google-spreadsheet');
    }
    return config;
  },
};

module.exports = nextConfig;
