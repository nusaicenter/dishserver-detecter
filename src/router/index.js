import { createRouter, createWebHashHistory } from 'vue-router'
import Home from '../views/Home.vue'

const routes = [
  {
    path: '/',
    redirect: 'dashboard'
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
        path: '/dashboard',
        name: 'dashboard',
        meta: {
          title: '菜品管理',
          icon: 'HomeFilled'
        },
        component: () =>
          import(/* webpackChunkName: "dashboard" */ '../views/Dashboard.vue')
      },
      {
        path: '/upload',
        name: 'upload',
        meta: {
          title: '菜品上传',
          icon: 'Upload'
        },
        component: () =>
          import(/* webpackChunkName: "upload" */ '../views/Upload.vue')
      },
      // {
      //   path: '/edit',
      //   name: 'edit',
      //   meta: {
      //     title: '菜品编辑',
      //     icon: 'Edit'
      //   },
      //   component: () =>
      //     import(/* webpackChunkName: "upload" */ '../views/Edit.vue')
      // },
      {
        path: '/user',
        name: 'user',
        meta: {
          title: '个人中心'
        },
        component: () =>
          import(/* webpackChunkName: "login" */ '../views/User.vue')
      }
    ]
  },
  {
    path: '/login',
    name: 'login',
    meta: {
      title: '登录'
    },
    component: () =>
      import(/* webpackChunkName: "login" */ '../views/Login.vue')
  },
  {
    path: '/register',
    name: 'register',
    meta: {
      title: '注册'
    },
    component: () =>
      import(/* webpackChunkName: "register" */ '../views/Register.vue')
  }
]

const router = createRouter({
  history: createWebHashHistory(),
  routes
})

export default router