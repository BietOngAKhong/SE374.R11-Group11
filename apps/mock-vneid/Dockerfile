FROM node:20-alpine

WORKDIR /app

COPY package.json ./
RUN npm install --omit=dev

COPY src ./src
COPY data ./data
COPY assets ./assets

EXPOSE 5001

ENV NODE_ENV=production
ENV PORT=5001
ENV HOST=0.0.0.0

CMD ["node", "src/index.js"]

