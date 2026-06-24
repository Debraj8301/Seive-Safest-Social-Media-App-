# Frontend Export

This folder contains the Seive React + Vite frontend prepared for cloud hosting.

## Netlify

- Base directory: leave empty if you upload this folder directly, or set it to `project_exports/Frontend` if deploying from the repo root
- Build command: `npm run build`
- Publish directory: `dist`

### Netlify Environment Variables

- `VITE_SUPABASE_URL=https://your-project.supabase.co`
- `VITE_SUPABASE_PUBLISHABLE_KEY=your-supabase-publishable-key`
- `VITE_MODERATION_API_URL=https://your-render-service.onrender.com`
- `VITE_BACKEND_API_URL=https://your-render-service.onrender.com`

## Included

- `src/`, `public/`, `supabase/`: source files
- `dist/`: latest production build
- `.env.example`: environment variables to set on your host
- `netlify.toml`: Netlify build and SPA redirect config

## Local Run

```bash
npm install
npm run dev
```

## Production Build

```bash
npm install
npm run build
```
