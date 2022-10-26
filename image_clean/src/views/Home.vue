<template>
  <div class="about">
    <v-header />
    <v-sidebar />
    <div class="content-box" :class="{ 'content-collapse': collapse }">
      <div class="content">
        <router-view v-slot="{ Component }">
          <transition name="move" mode="out-in">
            <component :is="Component" />
          </transition>
        </router-view>
      </div>
    </div>
  </div>
</template>
<script setup>
import { storeToRefs } from 'pinia'

import { useRootStore } from '../store/root.js'

import vHeader from '../components/Header.vue'
import vSidebar from '../components/Sidebar.vue'

const rootStore = useRootStore()
const { collapse } = storeToRefs(rootStore)
</script>
