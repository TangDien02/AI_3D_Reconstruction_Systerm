import { StyleSheet, Text, View } from 'react-native';

import { colors } from '../constants/theme';

type PipelineItemProps = {
  index: string;
  title: string;
  text: string;
};

export function PipelineItem({ index, title, text }: PipelineItemProps) {
  return (
    <View style={styles.item}>
      <Text style={styles.index}>{index}</Text>
      <View style={styles.copy}>
        <Text style={styles.title}>{title}</Text>
        <Text style={styles.text}>{text}</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  item: {
    flexDirection: 'row',
    gap: 12,
  },
  index: {
    width: 34,
    color: colors.green,
    fontSize: 13,
    fontWeight: '900',
  },
  copy: {
    flex: 1,
  },
  title: {
    color: '#1e2d25',
    fontSize: 15,
    fontWeight: '800',
  },
  text: {
    marginTop: 3,
    color: colors.mutedLight,
    fontSize: 13,
    lineHeight: 18,
  },
});
