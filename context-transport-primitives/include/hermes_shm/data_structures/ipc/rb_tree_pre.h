/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef HSHM_DATA_STRUCTURES_IPC_RB_TREE_PRE_H_
#define HSHM_DATA_STRUCTURES_IPC_RB_TREE_PRE_H_

#include "hermes_shm/memory/allocator/allocator.h"
#include "hermes_shm/types/atomic.h"

namespace hshm::ipc::pre {

/**
 * Color enumeration for Red-Black tree nodes
 */
enum class RBColor : uint8_t {
  RED = 0,
  BLACK = 1
};

/**
 * Red-Black tree node base class for preallocated tree
 *
 * This node structure is designed to be inherited by user-defined node types.
 * It maintains the RB tree properties and linkage without storing the key.
 * The derived NodeT class must provide:
 * - A 'key' member variable for ordering
 * - Comparison operators (operator<, operator>, operator==)
 */
class rb_node {
 public:
  OffsetPtr<>left_;            /**< Offset pointer to left child */
  OffsetPtr<>right_;           /**< Offset pointer to right child */
  OffsetPtr<>parent_;          /**< Offset pointer to parent node */
  RBColor color_;                 /**< Node color (RED or BLACK) */

  /**
   * Default constructor
   */
  HSHM_CROSS_FUN
  rb_node() : left_(OffsetPtr<>::GetNull()),
              right_(OffsetPtr<>::GetNull()),
              parent_(OffsetPtr<>::GetNull()),
              color_(RBColor::RED) {}

  /**
   * Check if this is a null node
   */
  HSHM_CROSS_FUN
  bool IsNull() const {
    return left_.IsNull() && right_.IsNull() && parent_.IsNull();
  }
};

/**
 * Red-Black tree for preallocated nodes
 *
 * This is a shared-memory compatible balanced binary search tree that
 * does not perform allocations. All nodes must be preallocated by the caller.
 *
 * The tree maintains RB tree invariants:
 * 1. Every node is either red or black
 * 2. The root is black
 * 3. All leaves (NULL) are black
 * 4. Red nodes have black children
 * 5. All paths from root to leaves have the same number of black nodes
 *
 * @tparam NodeT The node type that inherits from rb_node and provides:
 *               - A 'key' member variable for ordering
 *               - Comparison operators (operator<, operator>, operator==)
 * @tparam ATOMIC Whether to use atomic operations for thread-safety
 */
template<typename NodeT, bool ATOMIC = false>
class rb_tree {
 public:
  using KeyT = decltype(std::declval<NodeT>().key);  /**< Deduced key type from NodeT */

 private:
  opt_atomic<size_t, ATOMIC> size_;  /**< Number of nodes in the tree */
  OffsetPtr<NodeT> root_;            /**< Offset pointer to root node */

 public:
  /**
   * Default constructor
   */
  HSHM_CROSS_FUN
  rb_tree() : size_(0), root_(OffsetPtr<NodeT>::GetNull()) {}

  /**
   * Initialize the tree
   */
  HSHM_CROSS_FUN
  void Init() {
    size_.store(0);
    root_ = OffsetPtr<NodeT>::GetNull();
  }

  /**
   * Get the number of nodes in the tree
   */
  HSHM_CROSS_FUN
  size_t size() const {
    return size_.load();
  }

  /**
   * Check if the tree is empty
   */
  HSHM_CROSS_FUN
  bool empty() const {
    return size_.load() == 0;
  }

  /**
   * Get the root pointer (for debugging/inspection)
   */
  HSHM_CROSS_FUN
  OffsetPtr<NodeT> GetRoot() const {
    return root_;
  }

  /**
   * Emplace a preallocated node into the tree
   *
   * @param alloc Allocator for address translation
   * @param node Preallocated node to insert
   */
  template<typename AllocT>
  HSHM_CROSS_FUN
  void emplace(AllocT *alloc, FullPtr<NodeT> node) {
    node->color_ = RBColor::RED;
    node->left_ = OffsetPtr<>::GetNull();
    node->right_ = OffsetPtr<>::GetNull();
    node->parent_ = OffsetPtr<>::GetNull();

    // Standard BST insertion
    if (root_.IsNull()) {
      root_ = node.shm_.off_;
      node->color_ = RBColor::BLACK;
      size_.store(size_.load() + 1);
      return;
    }

    // Find insertion point
    OffsetPtr<NodeT> curr_off = root_;
    OffsetPtr<NodeT> parent_off = OffsetPtr<NodeT>::GetNull();

    while (!curr_off.IsNull()) {
      FullPtr<NodeT> curr(alloc, curr_off);
      parent_off = curr_off;

      if (*node < *curr) {
        curr_off = OffsetPtr<NodeT>(curr->left_);
      } else if (*node > *curr) {
        curr_off = OffsetPtr<NodeT>(curr->right_);
      } else {
        // Key already exists - don't insert duplicate
        return;
      }
    }

    // Insert node
    node->parent_ = parent_off.template Cast<void>();
    FullPtr<NodeT> parent(alloc, parent_off);

    if (*node < *parent) {
      parent->left_ = node.shm_.off_.template Cast<void>();
    } else {
      parent->right_ = node.shm_.off_.template Cast<void>();
    }

    size_.store(size_.load() + 1);

    // Fix RB tree properties
    FixInsert(alloc, node);
  }

  /**
   * Remove a node from the tree
   *
   * @param alloc Allocator for address translation
   * @param key Key of the node to remove
   * @return FullPtr to the removed node, or null if not found
   */
  template<typename AllocT>
  HSHM_CROSS_FUN
  FullPtr<NodeT> pop(AllocT *alloc, const KeyT &key) {
    if (root_.IsNull()) {
      return FullPtr<NodeT>::GetNull();
    }

    // Find the node to delete
    OffsetPtr<NodeT> node_off = FindNode(alloc, key);
    if (node_off.IsNull()) {
      return FullPtr<NodeT>::GetNull();
    }

    FullPtr<NodeT> node(alloc, node_off);
    FullPtr<NodeT> result = node;  // Save for return

    // Node to be deleted and node to replace it
    OffsetPtr<> replace_off;
    RBColor original_color = node->color_;

    // Track where the black node was actually removed (for FixDeleteFromParent)
    OffsetPtr<NodeT> deleted_parent;
    bool deleted_was_left;

    if (node->left_.IsNull()) {
      // Case 1: No left child
      replace_off = node->right_;
      deleted_parent = OffsetPtr<NodeT>(node->parent_);
      deleted_was_left = false;
      if (!deleted_parent.IsNull()) {
        FullPtr<NodeT> parent(alloc, deleted_parent);
        deleted_was_left = (parent->left_.load() == node_off.load());
      }
      Transplant(alloc, node_off, node->right_.template Cast<void>());
    } else if (node->right_.IsNull()) {
      // Case 2: No right child
      replace_off = node->left_;
      deleted_parent = OffsetPtr<NodeT>(node->parent_);
      deleted_was_left = false;
      if (!deleted_parent.IsNull()) {
        FullPtr<NodeT> parent(alloc, deleted_parent);
        deleted_was_left = (parent->left_.load() == node_off.load());
      }
      Transplant(alloc, node_off, node->left_.template Cast<void>());
    } else {
      // Case 3: Two children - find successor
      OffsetPtr<NodeT> successor_off = Minimum(alloc, node->right_);
      FullPtr<NodeT> successor(alloc, successor_off);
      original_color = successor->color_;
      replace_off = successor->right_;

      // Track where successor was originally (that's where black node is removed)
      deleted_parent = OffsetPtr<NodeT>(successor->parent_);
      deleted_was_left = false;
      if (!deleted_parent.IsNull()) {
        FullPtr<NodeT> parent(alloc, deleted_parent);
        deleted_was_left = (parent->left_.load() == successor_off.load());
      }

      if (successor->parent_.load() == node_off.load()) {
        // Successor is direct child
        if (!replace_off.IsNull()) {
          FullPtr<NodeT> replace(alloc, OffsetPtr<NodeT>(replace_off.load()));
          replace->parent_ = successor_off.template Cast<void>();
        }
        // When successor is direct child of node, the deleted position's parent becomes the successor itself after transplant
        deleted_parent = successor_off;
      } else {
        Transplant(alloc, successor_off, successor->right_.template Cast<void>());
        successor->right_ = node->right_;
        if (!successor->right_.IsNull()) {
          FullPtr<NodeT> right_child(alloc, OffsetPtr<NodeT>(successor->right_.load()));
          right_child->parent_ = successor_off.template Cast<void>();
        }
      }

      Transplant(alloc, node_off, successor_off.template Cast<void>());
      successor->left_ = node->left_;
      if (!successor->left_.IsNull()) {
        FullPtr<NodeT> left_child(alloc, OffsetPtr<NodeT>(successor->left_.load()));
        left_child->parent_ = successor_off.template Cast<void>();
      }
      successor->color_ = node->color_;
    }

    size_.store(size_.load() - 1);

    // Fix RB properties if we deleted a black node
    if (original_color == RBColor::BLACK) {
      if (!replace_off.IsNull()) {
        FixDelete(alloc, replace_off);
      } else if (!deleted_parent.IsNull() && size_.load() > 0) {
        // Deleted a black leaf - need to fix from parent
        // The "double-black" null is at the position where the black node was removed
        FixDeleteFromParent(alloc, deleted_parent, deleted_was_left);
      }
    }

    // Ensure root is black
    if (!root_.IsNull()) {
      FullPtr<NodeT> root_node(alloc, root_);
      root_node->color_ = RBColor::BLACK;
    }

    // Clear the removed node's pointers
    result->left_ = OffsetPtr<>::GetNull();
    result->right_ = OffsetPtr<>::GetNull();
    result->parent_ = OffsetPtr<>::GetNull();

    return result;
  }

  /**
   * Find a node by key
   *
   * @param alloc Allocator for address translation
   * @param key Key to search for
   * @return FullPtr to the node, or null if not found
   */
  template<typename AllocT>
  HSHM_CROSS_FUN
  FullPtr<NodeT> find(AllocT *alloc, const KeyT &key) const {
    OffsetPtr<NodeT> node_off = FindNode(alloc, key);
    if (node_off.IsNull()) {
      return FullPtr<NodeT>::GetNull();
    }
    return FullPtr<NodeT>(alloc, node_off);
  }

 private:
  /**
   * Find node by key (internal helper)
   */
  template<typename AllocT>
  HSHM_CROSS_FUN
  OffsetPtr<NodeT> FindNode(AllocT *alloc, const KeyT &key) const {
    OffsetPtr<NodeT> curr_off = root_;

    while (!curr_off.IsNull()) {
      FullPtr<NodeT> curr(alloc, curr_off);

      if (key < curr->key) {
        curr_off = OffsetPtr<NodeT>(curr->left_);
      } else if (key > curr->key) {
        curr_off = OffsetPtr<NodeT>(curr->right_);
      } else {
        return curr_off;
      }
    }

    return OffsetPtr<NodeT>::GetNull();
  }

  /**
   * Find minimum node in subtree
   */
  template<typename AllocT>
  HSHM_CROSS_FUN
  OffsetPtr<NodeT> Minimum(AllocT *alloc, OffsetPtr<> node_off) const {
    OffsetPtr<NodeT> curr_off(node_off);
    while (!curr_off.IsNull()) {
      FullPtr<NodeT> node(alloc, curr_off);
      if (node->left_.IsNull()) {
        break;
      }
      curr_off = OffsetPtr<NodeT>(node->left_);
    }
    return curr_off;
  }

  /**
   * Replace subtree rooted at u with subtree rooted at v
   */
  template<typename AllocT>
  HSHM_CROSS_FUN
  void Transplant(AllocT *alloc, OffsetPtr<NodeT> u_off, OffsetPtr<> v_off) {
    FullPtr<NodeT> u(alloc, u_off);

    if (u->parent_.IsNull()) {
      root_ = OffsetPtr<NodeT>(v_off);
    } else {
      FullPtr<NodeT> parent(alloc, OffsetPtr<NodeT>(u->parent_.load()));
      if (u_off.load() == parent->left_.load()) {
        parent->left_ = v_off;
      } else {
        parent->right_ = v_off;
      }
    }

    if (!v_off.IsNull()) {
      FullPtr<NodeT> v(alloc, OffsetPtr<NodeT>(v_off.load()));
      v->parent_ = u->parent_;
    }
  }

  /**
   * Left rotation
   */
  template<typename AllocT>
  HSHM_CROSS_FUN
  void RotateLeft(AllocT *alloc, OffsetPtr<NodeT> x_off) {
    FullPtr<NodeT> x(alloc, x_off);
    OffsetPtr<NodeT> y_off(x->right_);
    FullPtr<NodeT> y(alloc, y_off);

    x->right_ = y->left_;
    if (!y->left_.IsNull()) {
      FullPtr<NodeT> left_child(alloc, OffsetPtr<NodeT>(y->left_.load()));
      left_child->parent_ = x_off.template Cast<void>();
    }

    y->parent_ = x->parent_;
    if (x->parent_.IsNull()) {
      root_ = y_off;
    } else {
      FullPtr<NodeT> parent(alloc, OffsetPtr<NodeT>(x->parent_.load()));
      if (x_off.load() == parent->left_.load()) {
        parent->left_ = y_off.template Cast<void>();
      } else {
        parent->right_ = y_off.template Cast<void>();
      }
    }

    y->left_ = x_off.template Cast<void>();
    x->parent_ = y_off.template Cast<void>();
  }

  /**
   * Right rotation
   */
  template<typename AllocT>
  HSHM_CROSS_FUN
  void RotateRight(AllocT *alloc, OffsetPtr<NodeT> y_off) {
    FullPtr<NodeT> y(alloc, y_off);
    OffsetPtr<NodeT> x_off(y->left_);
    FullPtr<NodeT> x(alloc, x_off);

    y->left_ = x->right_;
    if (!x->right_.IsNull()) {
      FullPtr<NodeT> right_child(alloc, OffsetPtr<NodeT>(x->right_.load()));
      right_child->parent_ = y_off.template Cast<void>();
    }

    x->parent_ = y->parent_;
    if (y->parent_.IsNull()) {
      root_ = x_off;
    } else {
      FullPtr<NodeT> parent(alloc, OffsetPtr<NodeT>(y->parent_.load()));
      if (y_off.load() == parent->left_.load()) {
        parent->left_ = x_off.template Cast<void>();
      } else {
        parent->right_ = x_off.template Cast<void>();
      }
    }

    x->right_ = y_off.template Cast<void>();
    y->parent_ = x_off.template Cast<void>();
  }

  /**
   * Fix RB tree properties after insertion
   */
  template<typename AllocT>
  HSHM_CROSS_FUN
  void FixInsert(AllocT *alloc, FullPtr<NodeT> node) {
    OffsetPtr<NodeT> node_off = node.shm_.off_;

    while (!node->parent_.IsNull()) {
      FullPtr<NodeT> parent(alloc, OffsetPtr<NodeT>(node->parent_.load()));
      if (parent->color_ == RBColor::BLACK) {
        break;
      }

      if (parent->parent_.IsNull()) {
        break;
      }

      FullPtr<NodeT> grandparent(alloc, OffsetPtr<NodeT>(parent->parent_.load()));

      if (node->parent_.load() == grandparent->left_.load()) {
        // Parent is left child
        OffsetPtr<> uncle_off = grandparent->right_;

        if (!uncle_off.IsNull()) {
          FullPtr<NodeT> uncle(alloc, OffsetPtr<NodeT>(uncle_off.load()));
          if (uncle->color_ == RBColor::RED) {
            // Case 1: Uncle is red
            parent->color_ = RBColor::BLACK;
            uncle->color_ = RBColor::BLACK;
            grandparent->color_ = RBColor::RED;
            node_off = OffsetPtr<NodeT>(parent->parent_);
            node = FullPtr<NodeT>(alloc, node_off);
            continue;
          }
        }

        if (node_off.load() == parent->right_.load()) {
          // Case 2: Node is right child
          node_off = OffsetPtr<NodeT>(node->parent_.load());
          RotateLeft(alloc, node_off);
          node = FullPtr<NodeT>(alloc, node_off);
          parent = FullPtr<NodeT>(alloc, OffsetPtr<NodeT>(node->parent_.load()));
          grandparent = FullPtr<NodeT>(alloc, OffsetPtr<NodeT>(parent->parent_.load()));
        }

        // Case 3: Node is left child
        parent->color_ = RBColor::BLACK;
        grandparent->color_ = RBColor::RED;
        RotateRight(alloc, OffsetPtr<NodeT>(parent->parent_));
      } else {
        // Parent is right child (symmetric)
        OffsetPtr<> uncle_off = grandparent->left_;

        if (!uncle_off.IsNull()) {
          FullPtr<NodeT> uncle(alloc, OffsetPtr<NodeT>(uncle_off.load()));
          if (uncle->color_ == RBColor::RED) {
            parent->color_ = RBColor::BLACK;
            uncle->color_ = RBColor::BLACK;
            grandparent->color_ = RBColor::RED;
            node_off = OffsetPtr<NodeT>(parent->parent_);
            node = FullPtr<NodeT>(alloc, node_off);
            continue;
          }
        }

        if (node_off.load() == parent->left_.load()) {
          node_off = OffsetPtr<NodeT>(node->parent_.load());
          RotateRight(alloc, node_off);
          node = FullPtr<NodeT>(alloc, node_off);
          parent = FullPtr<NodeT>(alloc, OffsetPtr<NodeT>(node->parent_.load()));
          grandparent = FullPtr<NodeT>(alloc, OffsetPtr<NodeT>(parent->parent_.load()));
        }

        parent->color_ = RBColor::BLACK;
        grandparent->color_ = RBColor::RED;
        RotateLeft(alloc, OffsetPtr<NodeT>(parent->parent_));
      }
    }

    // Ensure root is black
    if (!root_.IsNull()) {
      FullPtr<NodeT> root(alloc, root_);
      root->color_ = RBColor::BLACK;
    }
  }

  /**
   * Fix RB tree properties after deletion
   */
  template<typename AllocT>
  HSHM_CROSS_FUN
  void FixDelete(AllocT *alloc, OffsetPtr<> node_off_raw) {
    OffsetPtr<NodeT> node_off(node_off_raw);
    while (node_off.load() != root_.load()) {
      FullPtr<NodeT> node(alloc, node_off);
      if (node->color_ == RBColor::RED) {
        break;
      }

      if (node->parent_.IsNull()) {
        break;
      }

      FullPtr<NodeT> parent(alloc, OffsetPtr<NodeT>(node->parent_.load()));

      if (node_off.load() == parent->left_.load()) {
        OffsetPtr<NodeT> sibling_off(parent->right_);
        if (sibling_off.IsNull()) {
          node_off = OffsetPtr<NodeT>(node->parent_.load());
          continue;
        }

        FullPtr<NodeT> sibling(alloc, sibling_off);

        if (sibling->color_ == RBColor::RED) {
          sibling->color_ = RBColor::BLACK;
          parent->color_ = RBColor::RED;
          RotateLeft(alloc, OffsetPtr<NodeT>(node->parent_));
          sibling_off = OffsetPtr<NodeT>(parent->right_.load());
          sibling = FullPtr<NodeT>(alloc, OffsetPtr<NodeT>(sibling_off.load()));
        }

        bool left_black = sibling->left_.IsNull();
        bool right_black = sibling->right_.IsNull();

        if (!sibling->left_.IsNull()) {
          FullPtr<NodeT> left(alloc, OffsetPtr<NodeT>(sibling->left_.load()));
          left_black = (left->color_ == RBColor::BLACK);
        }
        if (!sibling->right_.IsNull()) {
          FullPtr<NodeT> right(alloc, OffsetPtr<NodeT>(sibling->right_.load()));
          right_black = (right->color_ == RBColor::BLACK);
        }

        if (left_black && right_black) {
          sibling->color_ = RBColor::RED;
          node_off = OffsetPtr<NodeT>(node->parent_.load());
        } else {
          if (right_black) {
            if (!sibling->left_.IsNull()) {
              FullPtr<NodeT> left(alloc, OffsetPtr<NodeT>(sibling->left_.load()));
              left->color_ = RBColor::BLACK;
            }
            sibling->color_ = RBColor::RED;
            RotateRight(alloc, OffsetPtr<NodeT>(sibling_off));
            sibling_off = OffsetPtr<NodeT>(parent->right_.load());
            sibling = FullPtr<NodeT>(alloc, OffsetPtr<NodeT>(sibling_off.load()));
          }

          sibling->color_ = parent->color_;
          parent->color_ = RBColor::BLACK;
          if (!sibling->right_.IsNull()) {
            FullPtr<NodeT> right(alloc, OffsetPtr<NodeT>(sibling->right_.load()));
            right->color_ = RBColor::BLACK;
          }
          RotateLeft(alloc, OffsetPtr<NodeT>(node->parent_));
          node_off = root_;
        }
      } else {
        // Symmetric case
        OffsetPtr<NodeT> sibling_off(parent->left_);
        if (sibling_off.IsNull()) {
          node_off = OffsetPtr<NodeT>(node->parent_.load());
          continue;
        }

        FullPtr<NodeT> sibling(alloc, sibling_off);

        if (sibling->color_ == RBColor::RED) {
          sibling->color_ = RBColor::BLACK;
          parent->color_ = RBColor::RED;
          RotateRight(alloc, OffsetPtr<NodeT>(node->parent_));
          sibling_off = OffsetPtr<NodeT>(parent->left_.load());
          sibling = FullPtr<NodeT>(alloc, OffsetPtr<NodeT>(sibling_off.load()));
        }

        bool left_black = sibling->left_.IsNull();
        bool right_black = sibling->right_.IsNull();

        if (!sibling->left_.IsNull()) {
          FullPtr<NodeT> left(alloc, OffsetPtr<NodeT>(sibling->left_.load()));
          left_black = (left->color_ == RBColor::BLACK);
        }
        if (!sibling->right_.IsNull()) {
          FullPtr<NodeT> right(alloc, OffsetPtr<NodeT>(sibling->right_.load()));
          right_black = (right->color_ == RBColor::BLACK);
        }

        if (left_black && right_black) {
          sibling->color_ = RBColor::RED;
          node_off = OffsetPtr<NodeT>(node->parent_.load());
        } else {
          if (left_black) {
            if (!sibling->right_.IsNull()) {
              FullPtr<NodeT> right(alloc, OffsetPtr<NodeT>(sibling->right_.load()));
              right->color_ = RBColor::BLACK;
            }
            sibling->color_ = RBColor::RED;
            RotateLeft(alloc, OffsetPtr<NodeT>(sibling_off));
            sibling_off = OffsetPtr<NodeT>(parent->left_.load());
            sibling = FullPtr<NodeT>(alloc, OffsetPtr<NodeT>(sibling_off.load()));
          }

          sibling->color_ = parent->color_;
          parent->color_ = RBColor::BLACK;
          if (!sibling->left_.IsNull()) {
            FullPtr<NodeT> left(alloc, OffsetPtr<NodeT>(sibling->left_.load()));
            left->color_ = RBColor::BLACK;
          }
          RotateRight(alloc, OffsetPtr<NodeT>(node->parent_));
          node_off = root_;
        }
      }
    }

    if (!node_off.IsNull()) {
      FullPtr<NodeT> node(alloc, OffsetPtr<NodeT>(node_off.load()));
      node->color_ = RBColor::BLACK;
    }
  }

  /**
   * Fix RB properties when a black leaf was deleted (replacement is NULL)
   * This is called with the parent of the deleted node
   */
  template<typename AllocT>
  HSHM_CROSS_FUN
  void FixDeleteFromParent(AllocT *alloc, OffsetPtr<NodeT> parent_off, bool deleted_was_left) {
    while (true) {
      FullPtr<NodeT> parent(alloc, parent_off);

      if (deleted_was_left) {
        // Deleted node was left child
        OffsetPtr<NodeT> sibling_off(parent->right_);
        if (sibling_off.IsNull()) break;  // Shouldn't happen in valid RB tree

        FullPtr<NodeT> sibling(alloc, sibling_off);

        // Case 1: Red sibling
        if (sibling->color_ == RBColor::RED) {
          sibling->color_ = RBColor::BLACK;
          parent->color_ = RBColor::RED;
          RotateLeft(alloc, parent_off);
          sibling_off = OffsetPtr<NodeT>(parent->right_.load());
          if (sibling_off.IsNull()) break;
          sibling = FullPtr<NodeT>(alloc, OffsetPtr<NodeT>(sibling_off.load()));
        }

        // Check sibling's children colors
        bool left_black = sibling->left_.IsNull();
        bool right_black = sibling->right_.IsNull();
        if (!sibling->left_.IsNull()) {
          FullPtr<NodeT> left(alloc, OffsetPtr<NodeT>(sibling->left_.load()));
          left_black = (left->color_ == RBColor::BLACK);
        }
        if (!sibling->right_.IsNull()) {
          FullPtr<NodeT> right(alloc, OffsetPtr<NodeT>(sibling->right_.load()));
          right_black = (right->color_ == RBColor::BLACK);
        }

        // Case 2: Sibling and both nephews are black
        if (left_black && right_black) {
          sibling->color_ = RBColor::RED;
          if (parent->color_ == RBColor::RED) {
            parent->color_ = RBColor::BLACK;
            return;
          }
          // If parent is root, we're done (reduced black height of whole tree)
          if (parent_off.load() == root_.load()) {
            return;
          }
          // Continue fixing from parent
          if (parent->parent_.IsNull()) break;
          OffsetPtr<NodeT> grandparent_off(parent->parent_.load());
          FullPtr<NodeT> grandparent(alloc, grandparent_off);
          deleted_was_left = (grandparent->left_.load() == parent_off.load());
          parent_off = grandparent_off;
        } else {
          // Case 3: Right nephew is black (left is red)
          if (right_black) {
            if (!sibling->left_.IsNull()) {
              FullPtr<NodeT> left(alloc, OffsetPtr<NodeT>(sibling->left_.load()));
              left->color_ = RBColor::BLACK;
            }
            sibling->color_ = RBColor::RED;
            RotateRight(alloc, OffsetPtr<NodeT>(sibling_off));
            sibling_off = OffsetPtr<NodeT>(parent->right_.load());
            sibling = FullPtr<NodeT>(alloc, OffsetPtr<NodeT>(sibling_off.load()));
          }

          // Case 4: Right nephew is red
          sibling->color_ = parent->color_;
          parent->color_ = RBColor::BLACK;
          if (!sibling->right_.IsNull()) {
            FullPtr<NodeT> right(alloc, OffsetPtr<NodeT>(sibling->right_.load()));
            right->color_ = RBColor::BLACK;
          }
          RotateLeft(alloc, parent_off);
          return;
        }
      } else {
        // Deleted node was right child (symmetric)
        OffsetPtr<NodeT> sibling_off(parent->left_);
        if (sibling_off.IsNull()) break;

        FullPtr<NodeT> sibling(alloc, sibling_off);

        if (sibling->color_ == RBColor::RED) {
          sibling->color_ = RBColor::BLACK;
          parent->color_ = RBColor::RED;
          RotateRight(alloc, parent_off);
          sibling_off = OffsetPtr<NodeT>(parent->left_.load());
          if (sibling_off.IsNull()) break;
          sibling = FullPtr<NodeT>(alloc, OffsetPtr<NodeT>(sibling_off.load()));
        }

        bool left_black = sibling->left_.IsNull();
        bool right_black = sibling->right_.IsNull();
        if (!sibling->left_.IsNull()) {
          FullPtr<NodeT> left(alloc, OffsetPtr<NodeT>(sibling->left_.load()));
          left_black = (left->color_ == RBColor::BLACK);
        }
        if (!sibling->right_.IsNull()) {
          FullPtr<NodeT> right(alloc, OffsetPtr<NodeT>(sibling->right_.load()));
          right_black = (right->color_ == RBColor::BLACK);
        }

        if (left_black && right_black) {
          sibling->color_ = RBColor::RED;
          if (parent->color_ == RBColor::RED) {
            parent->color_ = RBColor::BLACK;
            return;
          }
          // If parent is root, we're done (reduced black height of whole tree)
          if (parent_off.load() == root_.load()) {
            return;
          }
          if (parent->parent_.IsNull()) break;
          OffsetPtr<NodeT> grandparent_off(parent->parent_.load());
          FullPtr<NodeT> grandparent(alloc, grandparent_off);
          deleted_was_left = (grandparent->left_.load() == parent_off.load());
          parent_off = grandparent_off;
        } else {
          if (left_black) {
            if (!sibling->right_.IsNull()) {
              FullPtr<NodeT> right(alloc, OffsetPtr<NodeT>(sibling->right_.load()));
              right->color_ = RBColor::BLACK;
            }
            sibling->color_ = RBColor::RED;
            RotateLeft(alloc, OffsetPtr<NodeT>(sibling_off));
            sibling_off = OffsetPtr<NodeT>(parent->left_.load());
            sibling = FullPtr<NodeT>(alloc, OffsetPtr<NodeT>(sibling_off.load()));
          }

          sibling->color_ = parent->color_;
          parent->color_ = RBColor::BLACK;
          if (!sibling->left_.IsNull()) {
            FullPtr<NodeT> left(alloc, OffsetPtr<NodeT>(sibling->left_.load()));
            left->color_ = RBColor::BLACK;
          }
          RotateRight(alloc, parent_off);
          return;
        }
      }
    }
  }

};

}  // namespace hshm::ipc::pre

#endif  // HSHM_DATA_STRUCTURES_IPC_RB_TREE_PRE_H_
