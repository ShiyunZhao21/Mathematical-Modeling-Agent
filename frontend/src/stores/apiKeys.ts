import { defineStore } from "pinia";
import { ref, computed } from "vue";
import { AgentType } from "@/utils/enum";
import type { ModelConfig } from "@/utils/interface";


export const useApiKeyStore = defineStore('apiKeys', () => {
  // 各个模型的配置
  const coordinatorConfig = ref<ModelConfig>({
    apiKey: '',
    baseUrl: '',
    modelId: '',
    provider: ''
  });

  const modelerConfig = ref<ModelConfig>({
    apiKey: '',
    baseUrl: '',
    modelId: '',
    provider: ''
  });

  const coderConfig = ref<ModelConfig>({
    apiKey: '',
    baseUrl: '',
    modelId: '',
    provider: ''
  });

  const writerConfig = ref<ModelConfig>({
    apiKey: '',
    baseUrl: '',
    modelId: '',
    provider: ''
  });

  const openalexEmail = ref<string>('');

  const isEmpty = computed(() => {
    return Object.values(getAllAgentConfigs()).every(config => config.apiKey === '')
  })

  // 设置协调者模型配置
  function setCoordinatorConfig(config: ModelConfig) {
    coordinatorConfig.value = { ...config };
  }

  // 设置建模者模型配置
  function setModelerConfig(config: ModelConfig) {
    modelerConfig.value = { ...config };
  }

  // 设置编码者模型配置
  function setCoderConfig(config: ModelConfig) {
    coderConfig.value = { ...config };
  }

  // 设置写作者模型配置
  function setWriterConfig(config: ModelConfig) {
    writerConfig.value = { ...config };
  }

  function setOpenalexEmail(email: string) {
    console.log('setOpenalexEmail', email)
    openalexEmail.value = email;
  }

  // 获取所有 agent 配置
  function getAllAgentConfigs() {
    return {
      [AgentType.COORDINATOR]: coordinatorConfig.value,
      [AgentType.MODELER]: modelerConfig.value,
      [AgentType.CODER]: coderConfig.value,
      [AgentType.WRITER]: writerConfig.value,
    };
  }

  // 重置所有配置
  function resetAll() {
    coordinatorConfig.value = { apiKey: '', baseUrl: '', modelId: '', provider: '' };
    modelerConfig.value = { apiKey: '', baseUrl: '', modelId: '', provider: '' };
    coderConfig.value = { apiKey: '', baseUrl: '', modelId: '', provider: '' };
    writerConfig.value = { apiKey: '', baseUrl: '', modelId: '', provider: '' };
    openalexEmail.value = '';
  }

  return {
    // 状态
    coordinatorConfig,
    modelerConfig,
    coderConfig,
    writerConfig,
    openalexEmail,
    isEmpty,

    // 方法
    setCoordinatorConfig,
    setModelerConfig,
    setCoderConfig,
    setWriterConfig,
    setOpenalexEmail,
    getAllAgentConfigs,
    resetAll
  }
}, {
  persist: true // 启用持久化存储
});
