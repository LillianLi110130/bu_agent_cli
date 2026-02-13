#!/usr/bin/env node

/**
 * API 类型生成脚本
 * 使用 swagger-typescript-api 从后端 OpenAPI 规范生成 TypeScript 类型定义
 */

import { generateApi } from 'swagger-typescript-api';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const swaggerUrl = 'http://localhost:8000/openapi.json';
// 输出相对于项目根目录的 src/api/types
const outputPath = path.resolve(__dirname, '../src/api/types');

async function main() {
  try {
    console.log('开始生成 API 类型...');
    console.log(`Swagger 文档地址: ${swaggerUrl}`);
    console.log(`输出目录: ${outputPath}`);

    await generateApi({
      url: swaggerUrl,
      output: outputPath,
      modular: true,
      cleanOutput: true,
      generateClient: false, // 不生成客户端代码，只生成类型
    });

    console.log('✅ API 类型生成完成！');
    console.log(`📁 生成目录: ${outputPath}`);
  } catch (error) {
    console.error('❌ API 类型生成失败:', error);
    process.exit(1);
  }
}

main();
