const value = process.env.VITE_API_URL
if (!value) throw new Error('Set VITE_API_URL to the deployed HTTPS backend origin in Vercel.')
const url = new URL(value)
if (url.protocol !== 'https:' || ['localhost', '127.0.0.1', '::1', '[::1]'].includes(url.hostname) || url.pathname !== '/' || url.search || url.hash || url.username || url.password) {
  throw new Error('VITE_API_URL must be a public HTTPS origin without a path, credentials, query or fragment.')
}
console.log(`Backend origin: ${url.origin}`)
