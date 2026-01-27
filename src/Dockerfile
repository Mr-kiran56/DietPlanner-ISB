# Build stage  Use Node 20
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package fil
COPY package*.json ./

# Install dependencies and vit
RUN npm cache clean --force && \
    npm install && \
    npm install -g vite @vitejs/plugin-react

# Copy all source 
COPY . .

# Verify installation and build
RUN which vite && \
    ls -la node_modules/.bin/ && \
    npm run build

# Production stag
FROM nginx:alpine

# Copy built assets from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx configuration (update path if needed)
COPY src/nginx.conf /etc/nginx/conf.d/default.conf


EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]