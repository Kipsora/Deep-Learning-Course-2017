import six


class HParamList(list):
    def __init__(self, hparams=None):
        super(HParamList, self).__init__(hparams)

    def __str__(self):
        result = ''
        for e in self:
            if result:
                result += ', '
            result += str(e)
        return self.__class__.__name__ + '(' + result + ')'

    def __repr__(self):
        return self.__str__()

    def iterall(self, prefix=None):
        for i, e in enumerate(self):
            if hasattr(e, 'iterall'):
                for k, v in e.iterall(prefix + '[{}]'.format(i)):
                    yield k, v
            else:
                yield prefix + '[{}]'.format(i), e


class HParam(dict):
    def __init__(self, hparam=None):
        super(HParam, self).__init__(hparam)

    def __getattr__(self, k):
        return super(HParam, self).get(k, None)

    def __str__(self):
        result = ''
        for k, v in six.iteritems(self):
            if result:
                result += ', '
            result += k + '=' + str(v)
        return self.__class__.__name__ + '(' + result + ')'

    def __repr__(self):
        return self.__str__()

    def __setattr__(self, key, value):
        if key not in self:
            raise AttributeError('Attribute \'{}\' is not found'.format(key))
        self[key] = get_incomplete_hparam(value)

    def iterall(self, prefix=''):
        if prefix:
            prefix += '.'
        for k, v in six.iteritems(self):
            if hasattr(v, 'iterall'):
                for k_, v_ in v.iterall(prefix + k):
                    yield k_, v_
            else:
                yield prefix + k, v

    def translate(self, d):
        result = dict()
        for k, v in six.iteritems(self):
            if k in d:
                result.setdefault(d[k], v)
            else:
                result.setdefault(k, v)
        return get_hparam(result)


def get_incomplete_hparam(hparam):
    if isinstance(hparam, dict):
        result = dict()
        for k, v in six.iteritems(hparam):
            result.setdefault(k, get_incomplete_hparam(v))
        return HParam(result)
    elif isinstance(hparam, list):
        result = list()
        for i, v in enumerate(hparam):
            result.append(get_incomplete_hparam(v))
        return HParamList(result)
    else:
        return hparam


def get_hparam(*args, **kwargs):
    tmp = dict()
    for arg in args:
        assert isinstance(arg, dict)
        for k, v in six.iteritems(arg):
            assert k not in tmp
            tmp.setdefault(k, get_incomplete_hparam(v))
    for k, v in six.iteritems(kwargs):
        assert k not in tmp
        tmp.setdefault(k, get_incomplete_hparam(v))
    return HParam(tmp)
