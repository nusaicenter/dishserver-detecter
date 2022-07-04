import { defineStore } from 'pinia'

export const useUserStore = defineStore('user', {
  state: () => {
    return {
      name: 'jessie'
    }
  },
  actions: {
    updateName(name) {
      this.name = name
    }
  }
})