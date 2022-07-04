<template>
  <div class="sidebar">
    <el-menu
      class="sidebar-el-menu"
      :default-active="onRoutes"
      :collapse="collapse"
      background-color="#324157"
      text-color="#bfcbd9"
      active-text-color="#20a0ff"
      unique-opened
      router
    >
      <template v-for="menu in menuList" :key="menu.index">
        <el-menu-item :index="menu.index">
          <el-icon>
            <component :is="menu.icon" />
          </el-icon>
          <template #title>{{ menu.name }}</template>
        </el-menu-item>
      </template>
    </el-menu>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { storeToRefs } from 'pinia'

import { useRootStore } from '../store/root'

const rootStore = useRootStore()
const { collapse } = storeToRefs(rootStore)

const menuList = [
  {
    index: '/dashboard',
    name: '菜品管理',
    icon: 'HomeFilled'
  },
  {
    index: '/upload',
    name: '菜品上传',
    icon: 'UploadFilled'
  },
  {
    index: '/user',
    name: '个人中心',
    icon: 'UserFilled'
  }
]

const route = useRoute()
// console.log('route', route)
const onRoutes = computed(() => {
  return route.path
})
</script>

<style scoped>
.sidebar {
  display: block;
  position: absolute;
  left: 0;
  top: 70px;
  bottom: 0;
  overflow-y: scroll;
}
.sidebar::-webkit-scrollbar {
  width: 0;
}
.sidebar-el-menu:not(.el-menu--collapse) {
  width: 250px;
}
.sidebar > ul {
  height: 100%;
}
</style>
