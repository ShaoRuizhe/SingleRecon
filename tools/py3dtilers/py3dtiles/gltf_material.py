import numpy as np
import sys


class GlTFMaterial():
    def __init__(self, metallicFactor=0, roughnessFactor=0, rgb=[1, 1, 1], alpha=1, textureUri=None):
        self.metallicFactor = metallicFactor
        self.roughnessFactor = roughnessFactor
        self.rgba = rgb
        if len(rgb) < 4:
            self.alpha = alpha
        self.textureUri = textureUri

    @property
    def metallicFactor(self):
        return self._metallicFactor

    @metallicFactor.setter
    def metallicFactor(self, value):
        self._metallicFactor = self.normalize_value(value, max(1, value))

    @property
    def roughnessFactor(self):
        return self._roughnessFactor

    @roughnessFactor.setter
    def roughnessFactor(self, value):
        self._roughnessFactor = self.normalize_value(value, max(1, value))

    @property
    def rgba(self):
        return self._rgba

    @rgba.setter
    def rgba(self, values):
        if len(values) < 3:
            for i in range(len(values), 3):
                values.append(0)
        elif len(values) > 4:
            values = values[:4]
        self._rgba = np.array(self.normalize_color(values), dtype=np.float32)

    @property
    def alpha(self):
        return self._rgba[3]

    @alpha.setter
    def alpha(self, value):
        value = self.normalize_value(value, self.max(value, 1, 255))
        if len(self._rgba) < 4:
            self._rgba = np.append(self._rgba, value)
        else:
            self._rgba[3] = value

    def is_textured(self):
        return self.textureUri is not None

    def to_dict(self, name, index=0):
        dictionary = {
            'pbrMetallicRoughness': {
                'baseColorFactor': self.rgba.tolist(),
                'metallicFactor': self.metallicFactor,
                'roughnessFactor': self.roughnessFactor
            },
            'name': name}
        if self.is_textured():
            dictionary['pbrMetallicRoughness']['baseColorTexture'] = {'index': index}
        return dictionary

    @staticmethod
    def from_hexa(color_code='#FFFFFF'):
        hex = color_code.replace('#', '').replace('0x', '')
        length = min(len(hex), 8)
        rgb = [round(int(hex[i:i + 2], 16) / 255, 4) for i in range(0, length, 2)]

        return GlTFMaterial(rgb=rgb)

    @classmethod
    def normalize_value(cls, value, max):
        if value < 0:
            print('The value can\'t be negative')
            sys.exit(1)
        return round(value / max, 4)

    @classmethod
    def max(cls, value, max_1, max_2):
        return max_1 if value <= max_1 else max(max_2, value)

    @classmethod
    def normalize_color(cls, color):
        max_value = cls.max(np.max(color), 1, 255)
        return [cls.normalize_value(value, max_value) for value in color]
