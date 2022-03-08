﻿#ifndef ACTION_H
#define ACTION_H

#include "Tile.h"

namespace_mahjong

enum class BaseAction : uint8_t {
	// response begin
	pass,
	吃, 
	碰,
	杠,
	荣和,
	// response end
	// 注意到所有的response Action可以通过大小来比较

	抢暗杠,
	抢杠,

	// self action begin
	暗杠,
	加杠,
	出牌,
	立直,
	自摸,
	九种九牌,
	// self action end
};

struct Action
{
	Action() = default;
	Action(BaseAction action, std::vector<Tile*>);
	BaseAction action = BaseAction::pass;

	/* 关于corresponding_tile的约定
	pass: empty (size=0)
	吃/碰: 手牌中2张牌 (size=2)
	杠: 手牌中2张牌 (size=3)
	荣/抢杠/抢暗杠: 荣到的那1张牌 (size=1)

	暗杠: 手牌中的4张牌 (size=4)
	加杠: 手牌中的1张牌 (size=1)
	出牌/立直: 手牌中的1张牌 (size=2)	
	自摸: empty (size=0)
	九种九牌: 能推的N种九牌，每种各一张 (size>=9)

	所有的corresponding_tile默认是排序的，否则判断吃牌会出现一定的问题
	*/
	std::vector<Tile*> correspond_tiles;
	std::string to_string() const;
    bool operator==(const Action& other) const;
	bool operator<(const Action& other) const;	
};

template<typename ActionType>
bool action_unique_pred(const ActionType& action1, const ActionType& action2)
{
	static_assert(std::is_base_of<Action, ActionType>::value, "Bad ActionType.");
	if (action1.action != action2.action) 
		return false;
	if (action1.correspond_tiles.size() != action2.correspond_tiles.size())
		return false;
	for (size_t i = 0; i < action1.correspond_tiles.size(); ++i) {
		if (action1.correspond_tiles[i]->red_dora ^ action2.correspond_tiles[i]->red_dora) {
			return false;
		}
		if (action1.correspond_tiles[i]->tile != action2.correspond_tiles[i]->tile) {
			return false;
		}
	}
	return true;
}

struct SelfAction : public Action
{
	SelfAction() = default;
	SelfAction(BaseAction, std::vector<Tile*>);
    bool operator==(const SelfAction& other);
};

template<typename ActionType>
int get_action_index(const std::vector<ActionType> &actions, BaseAction action_type, std::vector<BaseTile> correspond_tiles)
{
	// assume actions vector is sorted.
	int red_dora_match = -1;
	int idx = -1;

	switch (action_type) {
		case BaseAction::出牌:
		case BaseAction::立直:
			for (auto iter = actions.rbegin(); iter != actions.rend(); ++iter) {
				if (iter->action == action_type &&
					iter->correspond_tiles[0].tile == correspond_tiles[0])
				{
					// 倒序索引会优先打5保留0
					return actions.size() - 1 - (iter - actions.rbegin());
				}
			}
			break;
		case BaseAction::九种九牌:
			// 九种九牌就不看牌了
			for (auto iter = actions.begin(); iter != actions.end(); ++iter) {
				if (iter->action == action_type &&
					iter->correspond_tiles[0].tile == correspond_tiles[0])
				{
					// 倒序索引会优先打5保留0
					return iter - actions.rbegin();
				}
			}
			break;
		default: // 其他情况正序索引即可
			for (auto iter = actions.begin(); iter != actions.end(); ++iter) {
				if (iter->action == action_type &&
					iter->correspond_tiles.size() == correspond_tiles.size())
				{
					bool match = true;
					for (size_t i = 0; i< iter->correspond_tiles.size();++i){
						if (iter->correspond_tiles[i].tile != correspond_tiles[i]){
							match = false;
							break;
						}
					}
					if (match) return iter - actions.rbegin();
				}
			}
			break;
	}
	throw std::runtime_error("Cannot locate action.");
}

struct ResponseAction : public Action
{
	ResponseAction() = default;
	ResponseAction(BaseAction, std::vector<Tile*>);
    bool operator==(const ResponseAction& other);
};

namespace_mahjong_end

#endif