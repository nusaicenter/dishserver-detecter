import { defineStore } from 'pinia'

export const useRootStore = defineStore('root', {
  state: () => {
    return {
      collapse: false
    }
  },
  actions: {
    setCollapse(boolean) {
      this.collapse = boolean
    }
  }
})