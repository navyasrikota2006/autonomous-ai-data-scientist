# Stage 1: Build the React Application
FROM node:20-alpine AS build

WORKDIR /workspace

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

# Stage 2: Serve the Static Output via Nginx
FROM nginx:alpine

# Copy built app files to nginx default HTML directory
COPY --from=build /workspace/dist /usr/share/nginx/html

# Copy custom Nginx configuration to support Frontend Routing & Backend Proxying
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
