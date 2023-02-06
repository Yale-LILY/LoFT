"""Utility functions to interact with knowledge graph."""

import pprint
import collections

from nsm.execution.type_system import get_simple_type_hierarchy, DateTime


class Executor(object):
    """Executors implements the basic subroutines and provide
    the API to the computer.
    """

    def get_api(self, config):
        'Provide API to the computer.'
        raise NotImplementedError()


class SimpleKGExecutor(Executor):
    """This executor assumes that the knowledge graph is
    encoded as a dictionary.
    """

    def __init__(self, kg_info):
        """Given a knowledge graph, the number properties and
        the datetime properties, initialize an executor that
        implements the basic subroutines.

        Args:
          kg_info: a dictionary with three keys.
        """
        self.kg = kg_info['kg']
        self.num_props = kg_info['num_props']
        self.datetime_props = kg_info['datetime_props']
        self.props = kg_info['props']

    def hop(self, entities, prop, keep_dup=False):
        """Get the property of a list of entities."""
        if keep_dup:
            result = []
        else:
            result = set()
        for ent in entities:
            try:
                if keep_dup:
                    result += self.kg[ent][prop]
                else:
                    result = result.union(self.kg[ent][prop])
            except KeyError:
                continue
        return list(result)

    def filter_equal(self, ents_2, prop, ents_1):
        """From ents_1, filter out the entities whose property equal to ents_2."""
        result = []

        cast_func = self.get_cast_func(prop)
        query_ents = set(map(cast_func, ents_2))

        for ent in ents_1:
            if list(set(map(cast_func, self.hop([ent], prop)))) == list(set(query_ents)):
                result.append(ent)

        return result

    def filter_not_equal(self, ents_2, prop, ents_1):
        """From ents_1, filter out the entities whose property equal to ents_2."""
        result = []

        cast_func = self.get_cast_func(prop)
        query_ents = set(map(cast_func, ents_2))

        for ent in ents_1:
            if list(set(map(cast_func, self.hop([ent], prop)))) != list(set(query_ents)):
                result.append(ent)

        return result

    def get_cast_func(self, prop):
        if prop in self.datetime_props:
            return DateTime.from_string
        elif prop in self.num_props:
            return float
        else:
            return lambda x: x

        # raise RuntimeError('Not a valid ordering property [{}]'.format(prop))

    def get_num_prop_val(self, ent, prop):
        """Get the value of an entities' number property. """
        # If there are multiple values, then take the first one.
        prop_str_list = self.hop([ent], prop)
        try:
            prop_str = prop_str_list[0]
            prop_val = float(prop_str)
        except (ValueError, IndexError):
            prop_val = None
        return prop_val

    def get_datetime_prop_val(self, ent, prop):
        """Get the value of an entities' date time property. """
        # If there are multiple values, then take the first one.
        prop_str_list = self.hop([ent], prop)
        try:
            prop_str = prop_str_list[0]
            if prop_str[0] == '-':
                sign = -1
                prop_str = prop_str[1:]
            else:
                sign = 1
            result = [float(n) for n in prop_str.replace('x', '0').split('-')]
            day = 0
            for n, unit in zip(result, [365, 30, 1]):
                day += n * unit
            day *= sign
            prop_val = day
        except (ValueError, IndexError):
            prop_val = None
        return prop_val

    def sort_select(self, entities, prop, ind):
        """Sort the entities using prop then select the i-th one."""
        if prop in self.num_props:
            get_val = self.get_num_prop_val
        elif prop in self.datetime_props:
            get_val = self.get_datetime_prop_val
        else:
            raise (ValueError(prop))
        vals = []
        new_ents = []
        for ent in entities:
            val = get_val(ent, prop)
            if val is not None:
                new_ents.append(ent)
                vals.append(val)
        ent_vals = list(zip(new_ents, vals))
        assert len(new_ents) == len(vals)
        if len(vals) == 0:
            return []

        best_ent_val = sorted(
            ent_vals,
            key=lambda x: x[1])[ind]
        best_score = best_ent_val[1]
        result = [ent for ent, val in ent_vals if val == best_score]
        return result

    def argmax(self, entities, prop):
        return self.sort_select(entities, prop, -1)

    def argmin(self, entities, prop):
        return self.sort_select(entities, prop, 0)

    def show_kg(self):
        return pprint.pformat(self.kg, indent=2)

    def is_connected(self, source_ents, target_ents, prop):
        cast_func = self.get_cast_func(prop)

        try:
            result = set(map(cast_func, self.hop(source_ents, prop))) == set(map(cast_func, target_ents))
        except:
            return False

        return result

    def get_props(
            self, source_ents, target_ents=None, debug=False, condition_fn=None):
        """Get the properties that goes from source to targets."""
        props = set()
        if condition_fn is None:
            condition_fn = self.is_connected
        if debug:
            print('=' * 100)
        for ent in source_ents:
            if debug:
                print('@' * 20)
                print(ent)
            if ent in self.kg:
                ent_props = self.kg[ent].keys()
                if target_ents is not None:
                    for p in ent_props:
                        if debug:
                            print()
                            print(p)
                            print()
                            self.hop([ent], p)
                        # if set(self.hop([ent], p)) == set(target_ents):
                        if condition_fn([ent], target_ents, p):
                            props.add(p)
                else:
                    props = props.union(ent_props)
        if debug:
            print('in get props')
            print(source_ents)
            print(target_ents)
            print(props)
            print('=' * 100)

        return list(props)

    def autocomplete_argm(self, exp, tokens, token_vals, debug=False):
        l = len(exp)
        valid_tks = []
        if l == 1:  # first argument has more than one entity. ? or equal to one?
            valid_tks = [tk for tk, val in zip(tokens, token_vals)
                         if len(val['value']) >= 1 and not val['used']] + ['all_rows']
        elif l == 2:  # second argument is a property, and at least one ent contains it.
            ents = exp[1]['value']
            for tk, val in zip(tokens, token_vals):
                for ent in ents:
                    if val['value'] in self.kg[ent]:
                        valid_tks.append(tk)
                        break
        else:
            valid_tks = tokens
        if debug:
            print('*' * 30)
            print(exp)
            print(tokens)
            print(valid_tks)
            print('*' * 30)

        return valid_tks

    def autocomplete_filter_ops(self, exp, tokens, token_vals, debug=False):
        l = len(exp)
        if l == 1:
            # query entity -> num_list/datetime_list/string_list
            valid_tks = [tk for tk, val in zip(tokens, token_vals) if val['is_constant']]
        elif l == 2:
            # query property, must be type consistent with entity -> num_property/string_property/datetime_property
            valid_tks = [tk for tk, val in zip(tokens, token_vals) if val['type'][:3] == exp[1]['type'][:3]]
        elif l == 3:
            # [entity_list]
            valid_tks = [tk for tk, val in zip(tokens, token_vals) \
                if exp[2]['value'] not in val['prop']]
            valid_tks += ['all_rows']
        else:
            raise ValueError('Expression is too long: {}'.format(l))

        if debug:
            print()
            print('+' * 30)
            print('in filter equal')
            print(exp)
            print(tokens)
            print(valid_tks)
            print('+' * 30)

        return valid_tks

    def get_api(self):
        func_dict = collections.OrderedDict()
        # func_dict['hop'] = dict(
        #     name='hop',
        #     args=[{'types': ['entity_list']},
        #           {'types': ['property']}],
        #     return_type='entity_list',
        #     autocomplete=self.autocomplete_hop,
        #     value=self.hop)

        func_dict['filter_equal'] = dict(
            name='filter_equal',
            args=[{'types': ['entity_list']},
                  {'types': ['property']},
                  {'types': ['entity_list']}],
            return_type='entity_list',
            autocomplete=self.autocomplete_filter_ops,
            value=self.filter_equal)

        func_dict['argmax'] = dict(
            name='argmax',
            args=[{'types': ['entity_list']},
                  {'types': ['ordered_property']}],
            return_type='entity_list',
            autocomplete=self.autocomplete_argm,
            value=self.argmax)

        func_dict['argmin'] = dict(
            name='argmin',
            args=[{'types': ['entity_list']},
                  {'types': ['ordered_property']}],
            return_type='entity_list',
            autocomplete=self.autocomplete_argm,
            value=self.argmin)

        constant_dict = collections.OrderedDict()

        for p in self.props:
            if p in self.num_props:
                tp = 'num_property'
            elif p in self.datetime_props:
                tp = 'datetime_property'
            else:
                tp = 'string_property'

            constant_dict[p] = dict(
                value=p, type=tp, name=p)

        type_hierarchy = get_simple_type_hierarchy()
        return dict(type_hierarchy=type_hierarchy,
                    func_dict=func_dict,
                    constant_dict=constant_dict)


class TableExecutor(SimpleKGExecutor):
    """The executor for writing programs that processes simple Tables."""

    def __init__(self, table_info):
        super(TableExecutor, self).__init__(table_info)
        self.n_rows = len(table_info['row_ents'])

    def row_comparation(self, ents_1, ents_2, prop, operator='ge'):
        """return the comparation result between two rows"""
        cast_func = self.get_cast_func(prop)

        assert len(ents_1) == 1
        assert len(ents_2) == 1

        prop_values = list(map(
            cast_func,
            self.hop(ents_1, prop, keep_dup=True)
        ))[0], list(map(
            cast_func,
            self.hop(ents_2, prop, keep_dup=True)
        ))[0]

        if operator == 'greater':
            if prop_values[0] > prop_values[1]:
                return [True]
        elif operator == 'ge':
            if prop_values[0] >= prop_values[1]:
                return [True]
        elif operator == 'less':
            if prop_values[0] < prop_values[1]:
                return [True]
        elif operator == 'le':
            if prop_values[0] <= prop_values[1]:
                return [True]
        elif operator == 'same':
            if prop_values[0] == prop_values[1]:
                return [True]
        else:
            raise ValueError('Unknown operator {}'.format(operator))

        return [False]

    def row_greater(self, ents_1, ents_2, prop):
        return self.row_comparation(
            ents_1, ents_2, prop, operator='greater')

    def row_ge(self, ents_1, ents_2, prop):
        return self.row_comparation(
            ents_1, ents_2, prop, operator='ge')

    def row_less(self, ents_1, ents_2, prop):
        return self.row_comparation(
            ents_1, ents_2, prop, operator='less')

    def row_le(self, ents_1, ents_2, prop):
        return self.row_comparation(
            ents_1, ents_2, prop, operator='le')

    def same(self, ents_1, ents_2, prop):
        return self.row_comparation(
            ents_1, ents_2, prop, operator='same')

    def diff(self, ents_1, ents_2, prop):
        """Return the difference of two entities in prop."""
        assert len(ents_1) == 1
        assert len(ents_2) == 1
        val_1 = self.hop(ents_1, prop)[0]
        val_2 = self.hop(ents_2, prop)[0]
        return [abs(val_1 - val_2)]

    def autocomplete_row_comparation(self, exp, tokens, token_vals):
        """Autocomplete for diff function."""
        l = len(exp)
        if l == 1:
            valid_tks = [
                tk for tk, val in zip(tokens, token_vals)
                if len(val['value']) == 1 and not val['used']]
            # There must be at least two valid variables to apply diff.
            if len(valid_tks) < 2:
                valid_tks = []
        elif l == 2:
            valid_tks = [
                tk for tk, val in zip(tokens, token_vals)
                if len(val['value']) == 1 and val['name'] != exp[1]['name'] and not val['used']]
        else:
            props = set(self.get_props(exp[1]['value']))
            props = props.intersection(self.get_props(exp[2]['value']))
            valid_tks = []
            for tk, val in zip(tokens, token_vals):
                if val['value'] in props and val['value'] not in exp[1]['prop'] and val['value'] not in exp[2]['prop']:
                    valid_tks.append(tk)
        return valid_tks

    def filter_ge(self, nums, prop, ents_1):
        """Filter out entities whose prop >= nums."""
        result = []
        cast_func = self.get_cast_func(prop)

        casted_query_ents = [cast_func(x) for x in nums]
        for ent in ents_1:
            vals = set(map(cast_func, self.hop([ent], prop)))
            for val in vals:
                if all([(val >= x) for x in casted_query_ents]):
                    result.append(ent)
                    break
        return result

    def filter_greater(self, nums, prop, ents_1):
        """Filter out entities whose prop > nums."""
        result = []
        cast_func = self.get_cast_func(prop)

        casted_query_ents = [cast_func(x) for x in nums]
        for ent in ents_1:
            vals = set(map(cast_func, self.hop([ent], prop)))
            for val in vals:
                if all([(val > x) for x in casted_query_ents]):
                    result.append(ent)
                    break
        return result

    def filter_le(self, nums, prop, ents_1):
        """Filter out entities whose prop <= nums."""
        result = []
        cast_func = self.get_cast_func(prop)

        casted_query_ents = [cast_func(x) for x in nums]
        for ent in ents_1:
            vals = set(map(cast_func, self.hop([ent], prop)))
            for val in vals:
                if all([(val <= x) for x in casted_query_ents]):
                    result.append(ent)
                    break

        return result

    def filter_less(self, nums, prop, ents_1):
        """Filter out entities whose prop < nums."""
        result = []
        cast_func = self.get_cast_func(prop)

        casted_query_ents = [cast_func(x) for x in nums]
        for ent in ents_1:
            vals = set(map(cast_func, self.hop([ent], prop)))
            for val in vals:
                if all([(val < x) for x in casted_query_ents]):
                    result.append(ent)
                    break

        return result

    def filter_str_contain_any(self, string_list, prop, ents):
        """Filter out entities whose prop contains any of the strings."""
        result = []
        for ent in ents:
            if prop not in self.kg[ent]:
                continue
            str_val = self.hop([ent], prop)
            assert len(str_val) == 1
            for string in string_list:
                if string in str_val[0]:
                    result.append(ent)
                    break
        return result

    def filter_str_contain_not_any(self, string_list, prop, ents):
        """Filter out entities, whose prop doesn't contain any of the strings."""
        result = []
        for ent in ents:
            if prop not in self.kg[ent]:
                result.append(ent)
                continue
            str_val = self.hop([ent], prop)
            # Make sure that entity only has one value for the prop.
            assert len(str_val) == 1
            # If any one of the string is contained by the cell,
            # then pass. Only add to the result when none of the
            # string is in the cell.
            for string in string_list:
                if string in str_val:
                    break
            else:
                result.append(ent)
        return result

    def autocomplete_filter_str_contain_any(
            self, exp, tokens, token_vals, debug=False):
        """Auto-complete for filter_str_contain_any function."""
        l = len(exp)
        if l == 1:
            valid_tks = [tk for tk, val in zip(tokens, token_vals) if val['is_constant']]
        elif l == 2:
            # the prop is corresponding to the string, this is ensured by preprocess
            valid_tks = tokens
        elif l == 3:
            filter_func = None
            if exp[0] == 'filter_str_contain_any':
                filter_func = self.filter_str_contain_any
            else:
                filter_func = self.filter_str_contain_not_any
            valid_tks = [tk for tk, val in zip(tokens, token_vals) \
                if exp[2]['value'] not in val['prop'] or filter_func(exp[1]['value'], exp[2]['value'], val['value'])]
            valid_tks += ['all_rows']
        else:
            raise ValueError('Expression is too long: {}'.format(l))

        if debug:
            print()
            print('+' * 30)
            print('in filter equal')
            print(exp)
            print(tokens)
            print(valid_tks)
            print('+' * 30)

        return valid_tks

    # Next and previous
    def next(self, rows):
        """Select all the rows that is right below the given rows respectively."""
        assert rows
        assert rows[0][:4] == 'row_'
        # row are in the pattern of row_0, row_1.
        row_ids = [int(row_str[4:]) for row_str in rows]
        new_row_ids = [(i + 1) for i in row_ids if i + 1 < self.n_rows]
        if new_row_ids:
            result_rows = ['row_{}'.format(i) for i in new_row_ids]
            # result_rows = ['row_{}'.format(max(new_row_ids))]
        else:
            result_rows = []
        return result_rows

    def previous(self, rows):
        """Select all the rows that is right above the given rows respectively."""
        assert rows
        assert rows[0][:4] == 'row_'
        row_ids = [int(row_str[4:]) for row_str in rows]
        new_row_ids = [(i - 1) for i in row_ids if i - 1 >= 0]
        if new_row_ids:
            result_rows = ['row_{}'.format(i) for i in new_row_ids]
            # result_rows = ['row_{}'.format(min(new_row_ids))]
        else:
            result_rows = []
        return result_rows

    def autocomplete_next(self, exp, tokens, token_vals):
        """Autocompletion for next function."""
        l = len(exp)
        token_vals = [x['value'] for x in token_vals]
        if l == 1:
            # If there are any non-empty result, then it is available.
            valid_tks = []
            for tk, val in zip(tokens, token_vals):
                if len(val) > 0 and tk != 'all_rows' and self.next(val):
                    valid_tks.append(tk)
        else:
            raise ValueError('Wrong length: {}.'.format(l))
        return valid_tks

    def autocomplete_previous(self, exp, tokens, token_vals):
        """Autocompletion for previous function."""
        l = len(exp)
        token_vals = [x['value'] for x in token_vals]
        if l == 1:
            # If there are any non-empty result, then it is available.
            valid_tks = []
            for tk, val in zip(tokens, token_vals):
                if len(val) > 0 and tk != 'all_rows' and self.previous(val):
                    valid_tks.append(tk)
        else:
            raise ValueError('Wrong length: {}.'.format(l))
        return valid_tks

        # First and last

    def first(self, rows):
        """Take the first row (the one with minimum index) in all the rows."""
        assert len(rows) > 1
        assert rows[0][:4] == 'row_'
        # Return the row with the smallest id.
        row_ids = [int(row_str[4:]) for row_str in rows]
        result_row_id = min(row_ids)
        result_rows = ['row_{}'.format(result_row_id)]
        return result_rows

    def last(self, rows):
        """Take the last row (the one with maximum index) in all the rows."""
        assert len(rows) > 1
        assert rows[0][:4] == 'row_'
        # Return the row with the largest id.
        row_ids = [int(row_str[4:]) for row_str in rows]
        result_row_id = max(row_ids)
        result_rows = ['row_{}'.format(result_row_id)]
        return result_rows

    def autocomplete_first_last(self, exp, tokens, token_vals):
        """Autocompletion for both first and last."""
        l = len(exp)
        token_vals = [x['value'] for x in token_vals]
        if l == 1:
            # Only use first or last when you have more than one
            # entity.
            valid_tks = [tk for tk, val in zip(tokens, token_vals) if len(val) > 1]
        else:
            raise ValueError('Wrong length: {}.'.format(l))
        return valid_tks

        # Aggregation functions.

    def count(self, ents):
        return [len(ents)]

    def autocomplete_count(self, exp, tokens, token_vals):
        return tokens + ['all_rows']

    def sum(self, ents, prop):
        vals = self.hop(ents, prop, keep_dup=True)
        return [sum(vals)]

    def average(self, ents, prop):
        vals = self.hop(ents, prop, keep_dup=True)
        return [float(sum(vals)) / len(vals)]

    # ++++++++++++++++ add compare ops +++++++++++++++
    def eq(self, ol1, ol2):
        assert len(ol1) == 1
        assert len(ol2) == 1
        return [ol1[0] == ol2[0]]

    def ge(self, ol1, ol2):
        assert len(ol1) == 1
        assert len(ol2) == 1
        return [ol1[0] >= ol2[0]]

    def greater(self, ol1, ol2):
        assert len(ol1) == 1
        assert len(ol2) == 1
        return [ol1[0] > ol2[0]]

    def le(self, ol1, ol2):
        return [not self.greater(ol1, ol2)]

    def less(self, ol1, ol2):
        return [not self.ge(ol1, ol2)]

    def and_op(self, ol1, ol2):
        return [ol1[0] and ol2[0]]

    def autocomplete_ent_comparation(self, exp, tokens, token_vals):
        l = len(exp)
        if l == 1:
            valid_tks = [
                tk for tk, val in zip(tokens, token_vals) 
                if len(val['value']) == 1]
            if len(valid_tks) < 2:
                valid_tks = []
        elif l == 2:
            valid_tks = [
                tk for tk, val in zip(tokens, token_vals) 
                if len(val['value']) == 1 and val['type'] == exp[1]['type'] and \
                    val['name'] != exp[1]['name'] and \
                    (val['is_constant'] + exp[1]['is_constant'] <= 1) and\
                    not ('count' in val and exp[1]['value'][0] > 50) and\
                    not ('count' in exp[1] and val['value'][0] > 50)
            ]
        return valid_tks

    def is_none(self, ents):
        return [len(ents) == 0]

    def autocomplete_is_none(self, exp, tokens, token_vals):
        return [tk for tk, val in zip(tokens, token_vals) if not val['used']]

    def is_not(self, fact):
        return [not fact[0]]

    def maximum(self, big_scope, small_scope, prop):
        if len(small_scope) == 0:
            return [False]
        val = list(set(self.hop(small_scope, prop)))
        assert len(val) == 1
        return [max(self.hop(big_scope, prop)) == val[0]]

    def minimum(self, big_scope, small_scope, prop):
        if len(small_scope) == 0:
            return [False]
        val = list(set(self.hop(small_scope, prop)))
        assert len(val) == 1
        return [min(self.hop(big_scope, prop)) == val[0]]

    def autocomplete_mmum(self, exp, tokens, token_vals):
        def all_eq(ents, prop):
            vals = self.hop(ents, prop)
            if len(vals) == len(ents) and len(set(vals)) == 1:
                return True
            return False

        l = len(exp)
        if l == 1:
            # big scope, must have more than one row
            valid_tks = [tk for tk, val in zip(tokens, token_vals) if len(val['value']) >= 1] + ['all_rows']
        elif l == 2:
            # small scope, must be filtered under big scope
            valid_tks = [tk for tk, val in zip(tokens, token_vals) if \
                set(exp[1]['prop']).issubset(set(val['prop'])) and \
                exp[1]['name'] != val['name'] and \
                not val['used']]
        elif l == 3:
            # prop
            valid_tks = []
            if len(exp[2]['value']) == 0:
                valid_tks = tokens
            elif len(exp[2]['value']) == 1:
                props1 = self.get_props(exp[1]['value'])
                props2 = self.get_props(exp[2]['value'])
                valid_tks = [tk for tk, val in zip(tokens, token_vals) if val['value'] in props1 and val['value'] in props2]
            else:
                props1 = self.get_props(exp[1]['value'])
                valid_tks = [tk for tk, val in zip(tokens, token_vals) if val['value'] in props1 and all_eq(exp[2]['value'], val['value'])]
        return valid_tks

    def mode(self, ents1, ents2):
        """Return the value that appears the most in the prop of the entities."""
        return [(len(ents2) * 1. / len(ents1)) >= 0.5]

    def all(self, ents1, ents2):
        return [len(ents1) == len(ents2)]

    def only(self, ents1, ents2):
        return [len(ents2) == 1]
    #----------------------------------------------------------------------

    def return_all_tokens(self, unused_exp, tokens, unused_token_vals):
        return tokens

    def get_api(self):
        """Get the functions, constants and type hierarchy."""
        func_dict = collections.OrderedDict()

        def hop_return_type_fn(arg1_type, arg2_type):
            if arg2_type == 'num_property':
                return 'num_list'
            elif arg2_type == 'string_property':
                return 'string_list'
            elif arg2_type == 'datetime_property':
                return 'datetime_list'
            elif arg2_type == 'entity_property':
                return 'entity_list'
            else:
                raise ValueError('Unknown type {}'.format(arg2_type))

        # func_dict['hop'] = dict(
        #     name='hop',
        #     args=[{'types': ['entity_list']},
        #           {'types': ['property']}],
        #     return_type=hop_return_type_fn,
        #     autocomplete=self.autocomplete_hop,
        #     type='primitive_function',
        #     value=self.hop)

        func_dict['same'] = dict(
            name='same',
            args=[
                {'types': ['entity_list']},
                {'types': ['entity_list']},
                {'types': ['property']}
            ],
            return_type='bool',
            autocomplete=self.autocomplete_row_comparation,
            type='primitive_function',
            value=self.same
        )

        func_dict['row_greater'] = dict(
            name='row_greater',
            args=[
                {'types': ['entity_list']},
                {'types': ['entity_list']},
                {'types': ['ordered_property']}
            ],
            return_type='bool',
            autocomplete=self.autocomplete_row_comparation,
            type='primitive_function',
            value=self.row_greater
        )

        func_dict['row_ge'] = dict(
            name='row_ge',
            args=[
                {'types': ['entity_list']},
                {'types': ['entity_list']},
                {'types': ['ordered_property']}
            ],
            return_type='bool',
            autocomplete=self.autocomplete_row_comparation,
            type='primitive_function',
            value=self.row_ge
        )

        func_dict['row_less'] = dict(
            name='row_less',
            args=[
                {'types': ['entity_list']},
                {'types': ['entity_list']},
                {'types': ['ordered_property']}
            ],
            return_type='bool',
            autocomplete=self.autocomplete_row_comparation,
            type='primitive_function',
            value=self.row_less
        )

        func_dict['row_le'] = dict(
            name='row_le',
            args=[
                {'types': ['entity_list']},
                {'types': ['entity_list']},
                {'types': ['ordered_property']}
            ],
            return_type='bool',
            autocomplete=self.autocomplete_row_comparation,
            type='primitive_function',
            value=self.row_le
        )

        # Only use filter equal for number and date and
        # entities. Use filter_str_contain for string values.
        func_dict['filter_eq'] = dict(
            name='filter_eq',
            args=[{'types': ['ordered_list']},
                  {'types': ['ordered_property']},
                  {'types': ['entity_list']}],
            return_type='entity_list',
            autocomplete=self.autocomplete_filter_ops,
            type='primitive_function',
            value=self.filter_equal)

        func_dict['filter_not_eq'] = dict(
            name='filter_not_eq',
            args=[{'types': ['ordered_list']},
                  {'types': ['ordered_property']},
                  {'types': ['entity_list']}],
            return_type='entity_list',
            autocomplete=self.autocomplete_filter_ops,
            type='primitive_function',
            value=self.filter_not_equal)

        func_dict['argmax'] = dict(
            name='argmax',
            args=[{'types': ['entity_list']},
                  {'types': ['ordered_property']}],
            return_type='entity_list',
            autocomplete=self.autocomplete_argm,
            type='primitive_function',
            value=self.argmax)

        func_dict['argmin'] = dict(
            name='argmin',
            args=[{'types': ['entity_list']},
                  {'types': ['ordered_property']}],
            return_type='entity_list',
            autocomplete=self.autocomplete_argm,
            type='primitive_function',
            value=self.argmin)

        # func_dict['first'] = dict(
        #     name='first',
        #     args=[{'types': ['entity_list']}],
        #     return_type='entity_list',
        #     autocomplete=self.autocomplete_first_last,
        #     type='primitive_function',
        #     value=self.first)

        # func_dict['last'] = dict(
        #     name='last',
        #     args=[{'types': ['entity_list']}],
        #     return_type='entity_list',
        #     autocomplete=self.autocomplete_first_last,
        #     type='primitive_function',
        #     value=self.last)

        # func_dict['next'] = dict(
        #     name='next',
        #     args=[{'types': ['entity_list']}],
        #     return_type='entity_list',
        #     autocomplete=self.autocomplete_next,
        #     type='primitive_function',
        #     value=self.next)

        # func_dict['previous'] = dict(
        #     name='previous',
        #     args=[{'types': ['entity_list']}],
        #     return_type='entity_list',
        #     autocomplete=self.autocomplete_previous,
        #     type='primitive_function',
        #     value=self.previous)

        func_dict['filter_str_contain_any'] = dict(
            name='filter_str_contain_any',
            args=[{'types': ['string_list']},
                  {'types': ['string_property']},
                  {'types': ['entity_list']}],
            return_type='entity_list',
            autocomplete=self.autocomplete_filter_str_contain_any,
            type='primitive_function',
            value=self.filter_str_contain_any)

        func_dict['filter_str_contain_not_any'] = dict(
            name='filter_str_contain_not_any',
            args=[{'types': ['string_list']},
                  {'types': ['string_property']},
                  {'types': ['entity_list']}],
            return_type='entity_list',
            autocomplete=self.autocomplete_filter_str_contain_any,
            type='primitive_function',
            value=self.filter_str_contain_not_any)

        func_dict['filter_ge'] = dict(
            name='filter_ge',
            args=[{'types': ['ordered_list']},
                  {'types': ['ordered_property']},
                  {'types': ['entity_list']}],
            return_type='entity_list',
            autocomplete=self.autocomplete_filter_ops,
            type='primitive_function',
            value=self.filter_ge)

        func_dict['filter_greater'] = dict(
            name='filter_greater',
            args=[{'types': ['ordered_list']},
                  {'types': ['ordered_property']},
                  {'types': ['entity_list']}],
            return_type='entity_list',
            autocomplete=self.autocomplete_filter_ops,
            type='primitive_function',
            value=self.filter_greater)

        func_dict['filter_le'] = dict(
            name='filter_le',
            args=[{'types': ['ordered_list']},
                  {'types': ['ordered_property']},
                  {'types': ['entity_list']}],
            return_type='entity_list',
            autocomplete=self.autocomplete_filter_ops,
            type='primitive_function',
            value=self.filter_le)

        func_dict['filter_less'] = dict(
            name='filter_less',
            args=[{'types': ['ordered_list']},
                  {'types': ['ordered_property']},
                  {'types': ['entity_list']}],
            return_type='entity_list',
            autocomplete=self.autocomplete_filter_ops,
            type='primitive_function',
            value=self.filter_less)

        # aggregation functions.
        for k, f in zip(['maximum', 'minimum'],
                        [self.maximum, self.minimum]):
            func_dict[k] = dict(
                name=k,
                args=[{'types': ['entity_list']},
                      {'types': ['entity_list']},
                      {'types': ['ordered_property']}],
                # return_type='ordered_list',
                return_type='bool',
                autocomplete=self.autocomplete_mmum,
                type='primitive_function',
                value=f)

        func_dict['mode'] = dict(
            name='mode',
            args=[{'types': ['entity_list']},
                  {'types': ['entity_list']}],
            return_type='bool',
            autocomplete=self.autocomplete_mmum,
            type='primitive_function',
            value=self.mode)

        func_dict['only'] = dict(
            name='only',
            args=[{'types': ['entity_list']},
                  {'types': ['entity_list']}],
            return_type='bool',
            autocomplete=self.autocomplete_mmum,
            type='primitive_function',
            value=self.only)

        func_dict['all'] = dict(
            name='all',
            args=[{'types': ['entity_list']},
                  {'types': ['entity_list']}],
            return_type='bool',
            autocomplete=self.autocomplete_mmum,
            type='primitive_function',
            value=self.all)

        func_dict['count'] = dict(
            name='count',
            args=[{'types': ['entity_list']}],
            return_type='num_list',
            autocomplete=self.autocomplete_count,
            type='primitive_function',
            value=self.count)

        func_dict['average'] = dict(
            name='average',
            args=[{'types': ['entity_list']},
                  {'types': ['num_property']}],
            return_type='num_list',
            autocomplete=self.autocomplete_argm,
            type='primitive_function',
            value=self.average)

        func_dict['sum'] = dict(
            name='sum',
            args=[{'types': ['entity_list']},
                  {'types': ['num_property']}],
            return_type='num_list',
            autocomplete=self.autocomplete_argm,
            type='primitive_function',
            value=self.sum)

        func_dict['diff'] = dict(
            name='diff',
            args=[{'types': ['entity_list']},
                  {'types': ['entity_list']},
                  {'types': ['num_property']}],
            return_type='num_list',
            autocomplete=self.autocomplete_row_comparation,
            type='primitive_function',
            value=self.diff)

        #################### add func here #######################
        # in early version don't care about 'or' logic, and there are only 1% or in sentences 
        func_dict['is_none'] = dict(
            name='is_none',
            args=[{'types': ['entity_list']}],
            return_type='bool',
            autocomplete=self.autocomplete_is_none,
            type='primitive_function',
            value=self.is_none)

        func_dict['is_not'] = dict(
            name='is_not',
            args=[{'types': ['bool']}],
            return_type='bool',
            autocomplete=self.return_all_tokens,
            type='primitive_function',
            value=self.is_not)

        func_dict['eq'] = dict(
            name='eq',
            args=[{'types': ['ordered_list']},
                  {'types': ['ordered_list']}],
            return_type='bool',
            autocomplete=self.autocomplete_ent_comparation,
            type='primitive_function',
            value=self.eq)

        func_dict['ge'] = dict(
            name='ge',
            args=[{'types': ['ordered_list']},
                  {'types': ['ordered_list']}],
            return_type='bool',
            autocomplete=self.autocomplete_ent_comparation,
            type='primitive_function',
            value=self.ge)

        func_dict['greater'] = dict(
            name='greater',
            args=[{'types': ['ordered_list']},
                  {'types': ['ordered_list']}],
            return_type='bool',
            autocomplete=self.autocomplete_ent_comparation,
            type='primitive_function',
            value=self.greater
        )

        func_dict['le'] = dict(
            name='le',
            args=[{'types': ['ordered_list']},
                  {'types': ['ordered_list']}],
            return_type='bool',
            autocomplete=self.autocomplete_ent_comparation,
            type='primitive_function',
            value=self.le)

        func_dict['less'] = dict(
            name='less',
            args=[{'types': ['ordered_list']},
                  {'types': ['ordered_list']}],
            return_type='bool',
            autocomplete=self.autocomplete_ent_comparation,
            type='primitive_function',
            value=self.less
        )

        func_dict['and_op'] = dict(
            name='and_op',
            args=[{'types': ['bool']},
                  {'types': ['bool']}],
            return_type='bool',
            autocomplete=self.autocomplete_ent_comparation,
            type='primitive_function',
            value=self.and_op
        )
        #-------------------------------------------------------

        constant_dict = collections.OrderedDict()

        for p in self.props:
            if p in self.num_props:
                tp = 'num_property'
            elif p in self.datetime_props:
                tp = 'datetime_property'
            elif p.split('-')[-1] == 'entity':
                tp = 'entity_property'
            else:
                tp = 'string_property'

            constant_dict[p] = dict(
                value=p, type=tp, name=p)

        type_hierarchy = get_simple_type_hierarchy()
        return dict(type_hierarchy=type_hierarchy,
                    func_dict=func_dict,
                    constant_dict=constant_dict)
