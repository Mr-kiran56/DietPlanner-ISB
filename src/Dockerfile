# Build stage - Use Node 20
FROM node:20-alpine AS builder

# Create app directory with proper permissions
RUN mkdir -p /app && chown -R node:node /app
WORKDIR /app
USER node

# Copy package files
COPY --chown=node:node package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy all source files
COPY --chown=node:node . .

# Build the project
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built assets from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx configuration (update path if needed)
COPY src/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]