/* eslint-disable @typescript-eslint/no-explicit-any */
import axios, { type AxiosRequestConfig } from "axios";
import type {
  AxiosInstance,
  InternalAxiosRequestConfig,
  AxiosResponse,
} from "axios";

// 创建axios实例
const service: AxiosInstance = axios.create({
  timeout: 10000, // 请求超时时间
  headers: {
    "Content-Type": "application/json",
  },
});

// 请求拦截器
service.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    return config;
  },
  (error) => {
    // 对请求错误做些什么
    return Promise.reject(error);
  },
);

// 响应拦截器
service.interceptors.response.use(
  (response: AxiosResponse<any>) => {
      // 返回响应数据
      return response;
    },
  (error) => {
    // 对响应错误做点什么
    // 例如：处理HTTP状态码
    console.error(error.response);
    return Promise.reject(error);
  },
);

const request = <T = any>(config: AxiosRequestConfig): Promise<T> => {
  return service.request<any, T>(config);
};

export default request;
