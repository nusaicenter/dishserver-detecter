import { createRouter, createWebHashHistory } from 'vue-router'
import Home from '../views/Home.vue'

const routes = [
  {
    path: '/',
    redirect: 'videoprocess'
  },
  {
    path: '/',
    name: 'Home',
    component: Home,
    meta: {
      title: '菜品管理',
      icon: 'HomeFilled'
    },
    children: [
      {
        path: '/videoprocess',
        name: 'videoprocess',
        meta: {
          title: '视频处理'
        },
        component: () =>
          import(/* webpackChunkName: "login" */ '../views/VideoProcess.vue')
      },
      {
        path: '/mainimgclean',
        name: 'mainimgclean',
        meta: {
          title: '餐盘图片'
        },
        component: () =>
          import(/* webpackChunkName: "login" */ '../views/MainimgClean.vue')
      },
      {
        path: '/subimgclean',
        name: 'subimgclean',
        meta: {
          title: '菜品图片'
        },
        component: () =>
          import(/* webpackChunkName: "login" */ '../views/SubimgClean.vue')
      },
    ]
  }
]

const router = createRouter({
  history: createWebHashHistory(),
  routes
})

export default router