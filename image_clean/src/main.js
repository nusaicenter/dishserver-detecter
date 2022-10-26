import { createApp } from 'vue'

import App from './App.vue'
import router from './router'
import pinia from './store'

import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'

import * as Icons from '@element-plus/icons-vue'

const app = createApp(App)

for (const [key, component] of Object.entries(Icons)) {
  app.component(key, component)
}

app.use(router)
app.use(pinia)
app.use(ElementPlus)
app.mount('#app')
