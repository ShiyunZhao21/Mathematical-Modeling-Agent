<script setup lang="ts">
import { computed, ref, onMounted } from 'vue'
import Tree from '@/components/Tree.vue'
import { File } from 'lucide-vue-next'
import { useTaskStore } from '@/stores/task'
import { Sidebar, SidebarContent, SidebarGroup, SidebarFooter, SidebarProvider } from '@/components/ui/sidebar'
const taskStore = useTaskStore()
const isLoading = ref(true)
// 从消息中提取最新的文件列表
const files = taskStore.files as string[]

// 将文件列表转换为树形结构
const fileTree = computed(() => {
  // 无论files是否为空，只要计算属性被触发，就认为数据已加载完成
  isLoading.value = false

  // 直接返回文件列表，不做转换，因为Tree组件期望接收string或数组
  return files
})

// 添加超时机制，确保即使数据没有加载也会在一定时间后显示内容
onMounted(() => {
  // 3秒后无论如何都取消加载状态
  setTimeout(() => {
    isLoading.value = false
  }, 3000)
})

const handleFileClick = (file: string) => {
  // 处理文件点击
  console.log('File clicked:', file)
}

const handleFileDownload = (file: string) => {
  // 处理文件下载
  console.log('Download file:', file)
}
</script>

<template>
  <SidebarContent class="h-full">
    <SidebarGroup />
    <div class="h-full flex flex-col overflow-hidden">
      <div class="px-3 py-2 font-medium text-sm border-b">Files</div>
      <div class="flex-1 overflow-auto">
        <div v-if="isLoading" class="px-3 py-2 text-sm text-gray-500">
          加载中...
        </div>
        <div v-else-if="fileTree.length === 0" class="px-3 py-2 text-sm text-gray-500">
          暂无文件
        </div>
        <div v-else class="p-2">
          <Tree v-for="(item, index) in fileTree" :key="index" :item="item" @click="handleFileClick(item)"
            @download="handleFileDownload(item)" />
        </div>
      </div>
    </div>
    <SidebarGroup />
  </SidebarContent>
</template>
