#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

set -e

CANNBOT_URL="https://gitcode.com/cann/cannbot-skills.git"

# --- Color helpers ---
if [ -t 1 ]; then
    GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'
    CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; NC='\033[0m'
else
    GREEN=''; YELLOW=''; RED=''; CYAN=''; BOLD=''; DIM=''; NC=''
fi

ok()   { echo -e "  ${GREEN}✓${NC}${DIM} $*${NC}"; }
warn() { echo -e "  ${YELLOW}⚠${NC}${DIM} $*${NC}"; }
err()  { echo -e "  ${RED}✗${NC}${DIM} $*${NC}"; }
info() { echo -e "  ${CYAN}→${NC}${DIM} $*${NC}"; }
step() { echo -e "${DIM}$*${NC}"; }

VERSION="1.0.0"

show_help() {
    cat << EOF
ops-blas Agent 初始化脚本

Usage: bash init.sh <claude|opencode> [options]

Arguments:
  claude                    Target: Claude Code
  opencode                  Target: OpenCode

Options:
  --help, -h                Show this help message
  --clean                   Remove existing config directory before init
  --cannbot <path>          Path to cannbot-skills directory (default: clone from official)

Official URL:
  cannbot-skills: ${CANNBOT_URL}

Examples:
  bash init.sh claude
  bash init.sh opencode
  bash init.sh claude --clean
  bash init.sh claude --cannbot /path/to/cannbot-skills

What it does:
  1. Create config directory in ops-blas repo (.claude/ or .opencode/)
  2. Symlink agent/CLAUDE.md -> config/ (renamed to AGENTS.md for opencode)
  3. Symlink agent/agents/*.md -> config/agents/
  4. Setup cannbot-skills (use local path or clone from official)
  5. Symlink agent/skills/* -> config/skills/ (local skills)
  6. Read cannbot_references.json and symlink referenced cannbot skills
EOF
}

# --- Parse target environment ---
if [[ $# -lt 1 ]]; then
    echo -e "${RED}Error: Missing required argument <claude|opencode>${NC}"
    echo ""
    show_help
    exit 1
fi

case "$1" in
    --help|-h)
        show_help
        exit 0
        ;;
    claude|opencode)
        TARGET_ENV="$1"
        shift
        ;;
    *)
        echo -e "${RED}Error: First argument must be 'claude' or 'opencode'${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac

CLEAN_MODE=false
CANNBOT_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)
            show_help
            exit 0
            ;;
        --clean)
            CLEAN_MODE=true
            shift
            ;;
        --cannbot)
            if [ -z "$2" ] || [[ "$2" == --* ]]; then
                echo -e "${RED}Error: --cannbot requires a path argument${NC}"
                exit 1
            fi
            CANNBOT_PATH="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Error: Unknown argument '$1'${NC}"
            echo ""
            show_help
            exit 1
            ;;
    esac
done

# --- Environment-specific settings ---
if [ "$TARGET_ENV" = "claude" ]; then
    CONFIG_DIR_NAME=".claude"
    GUIDE_DST_NAME="CLAUDE.md"
    QUICK_START_CMD="cd $OPS_BLAS_DIR && claude"
else
    CONFIG_DIR_NAME=".opencode"
    GUIDE_DST_NAME="AGENTS.md"
    QUICK_START_CMD="cd $OPS_BLAS_DIR && opencode"
fi

# --- Resolve paths ---
SCRIPT_DIR=$(dirname "$(realpath "$0")")
AGENT_DIR="$SCRIPT_DIR"                          # agent/
OPS_BLAS_DIR=$(realpath "$SCRIPT_DIR/..")        # ops-blas/
CONFIG_DIR="$OPS_BLAS_DIR/$CONFIG_DIR_NAME"

if [ -n "$CANNBOT_PATH" ]; then
    CANNBOT_PATH=$(realpath "$CANNBOT_PATH" 2>/dev/null || echo "$CANNBOT_PATH")
    SKILLS_REPO="$CANNBOT_PATH"
else
    SKILLS_REPO="$CONFIG_DIR/ref-repos/cannbot-skills"
fi

# --- Display configuration ---
echo -e "  ${BOLD}Configuration:${NC}"
echo -e "  target env:   ${CYAN}$TARGET_ENV${NC} (config dir: $CONFIG_DIR_NAME/)"
echo -e "  ops-blas:     ${CYAN}$OPS_BLAS_DIR${NC}"
if [ -n "$CANNBOT_PATH" ]; then
    echo -e "  cannbot:      ${CYAN}$CANNBOT_PATH${NC} (local)"
else
    echo -e "  cannbot:      ${CYAN}clone from $CANNBOT_URL${NC}"
fi
echo ""

# --- Validate provided paths ---
if [ -n "$CANNBOT_PATH" ] && [ ! -d "$CANNBOT_PATH" ]; then
    err "cannbot-skills directory not found: $CANNBOT_PATH"
    exit 1
fi

if [ ! -d "$OPS_BLAS_DIR" ]; then
    err "ops-blas directory not found: $OPS_BLAS_DIR"
    exit 1
fi

cd "$OPS_BLAS_DIR"

# --- Step 1: Create config directory ---
if [ "$CLEAN_MODE" = true ] && [ -d "$CONFIG_DIR" ]; then
    info "Cleaning existing $CONFIG_DIR_NAME/ directory..."
    rm -rf "$CONFIG_DIR"
    ok "$CONFIG_DIR_NAME/ removed"
fi
step "[1/6] Creating $CONFIG_DIR_NAME directory..."
mkdir -p "$CONFIG_DIR"
ok "$CONFIG_DIR_NAME/ created"

# --- Step 2: Symlink agent/CLAUDE.md -> config/ (rename for opencode) ---
step "[2/6] Linking agent configuration..."
claude_md="$AGENT_DIR/CLAUDE.md"
if [ -f "$claude_md" ]; then
    dst="$CONFIG_DIR/$GUIDE_DST_NAME"
    if [ -L "$dst" ] || [ -e "$dst" ]; then
        rm -f "$dst"
    fi
    ln -sf "$claude_md" "$dst"
    ok "$GUIDE_DST_NAME -> agent/CLAUDE.md"
else
    warn "agent/CLAUDE.md not found, skipping"
fi

# --- Step 3: Symlink agent/agents/*.md -> config/agents/ ---
step "[3/6] Linking agents..."
mkdir -p "$CONFIG_DIR/agents"
local_agents="$AGENT_DIR/agents"
agent_count=0

if [ -d "$local_agents" ]; then
    for agent_file in "$local_agents"/*.md; do
        [ -f "$agent_file" ] || continue
        agent_name=$(basename "$agent_file")
        agent_dst="$CONFIG_DIR/agents/$agent_name"
        if [ -L "$agent_dst" ] || [ -e "$agent_dst" ]; then
            rm -f "$agent_dst"
        fi
        ln -sf "$agent_file" "$agent_dst"
        agent_count=$((agent_count + 1))
        ok "agent: $agent_name"
    done
    [ "$agent_count" -eq 0 ] && warn "No agents found in agent/agents/"
else
    warn "agent/agents/ directory not found, skipping"
fi

# --- Step 4: Setup cannbot-skills ---
step "[4/6] Setting up cannbot-skills..."

if [ -n "$CANNBOT_PATH" ]; then
    ok "Using local cannbot-skills: $CANNBOT_PATH"
else
    if [ -d "$SKILLS_REPO/.git" ]; then
        info "cannbot-skills already exists, updating..."
        cd "$SKILLS_REPO"
        git pull --quiet 2>/dev/null || warn "git pull failed, using existing version"
        cd "$OPS_BLAS_DIR"
        ok "cannbot-skills updated"
    else
        info "Cloning cannbot-skills from $CANNBOT_URL ..."
        git clone --quiet "$CANNBOT_URL" "$SKILLS_REPO" 2>/dev/null || {
            err "Failed to clone cannbot-skills from $CANNBOT_URL"
            exit 1
        }
        ok "cannbot-skills cloned"
    fi
fi

# --- Step 5: Symlink local skills -> config/skills/ ---
step "[5/6] Linking skills..."
mkdir -p "$CONFIG_DIR/skills"
local_skills="$AGENT_DIR/skills"
local_skill_count=0

if [ -d "$local_skills" ]; then
    for skill_dir in "$local_skills"/*; do
        [ -d "$skill_dir" ] || continue
        skill_name=$(basename "$skill_dir")
        if [ "$skill_name" = "cannbot_references.json" ]; then
            continue
        fi
        skill_dst="$CONFIG_DIR/skills/$skill_name"
        if [ -L "$skill_dst" ] || [ -e "$skill_dst" ]; then
            rm -rf "$skill_dst"
        fi
        ln -sf "$skill_dir" "$skill_dst"
        local_skill_count=$((local_skill_count + 1))
        ok "local skill: $skill_name"
    done
    [ "$local_skill_count" -eq 0 ] && warn "No local skills found in agent/skills/"
else
    warn "agent/skills/ directory not found, skipping"
fi

# --- Step 6: Link cannbot skills from cannbot_references.json ---
step "[6/6] Linking cannbot skills from cannbot_references.json..."
refs_json="$local_skills/cannbot_references.json"

if [ -f "$refs_json" ]; then
    cannbot_count=0
    cannbot_failed=0

    if command -v python3 &> /dev/null; then
        while IFS='|' read -r skill_name skill_path; do
            [ -z "$skill_name" ] && continue
            skill_src="$SKILLS_REPO/$skill_path"
            skill_dst="$CONFIG_DIR/skills/$skill_name"

            if [ -d "$skill_src" ]; then
                if [ -L "$skill_dst" ] || [ -e "$skill_dst" ]; then
                    rm -rf "$skill_dst"
                fi
                ln -sf "$skill_src" "$skill_dst"
                cannbot_count=$((cannbot_count + 1))
                ok "cannbot skill: $skill_name -> $skill_path"
            else
                warn "cannbot skill not found: $skill_name ($skill_path)"
                cannbot_failed=$((cannbot_failed + 1))
            fi
        done < <(python3 -c "
import json, sys
with open('$refs_json', 'r') as f:
    data = json.load(f)
for skill_name, paths in data.items():
    for p in paths:
        print(f'{skill_name}|{p}')
" 2>/dev/null) || warn "Failed to parse cannbot_references.json (empty or invalid JSON)"

        [ "$cannbot_count" -gt 0 ] && ok "Linked ${cannbot_count} cannbot skills"
        [ "$cannbot_failed" -gt 0 ] && warn "${cannbot_failed} cannbot skills not found"
    else
        warn "python3 not available, skipping cannbot skills linking"
    fi
else
    warn "cannbot_references.json not found, skipping cannbot skills linking"
fi

# --- Summary ---
echo ""
echo -e "  ${GREEN}${BOLD}✓ Initialization completed!${NC}"
echo ""
echo -e "  ${BOLD}Usage:${NC}"
echo -e "  $QUICK_START_CMD"
echo ""
